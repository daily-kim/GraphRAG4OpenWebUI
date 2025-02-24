import os
import asyncio
import time
import uuid
import json
import re
import pandas as pd
import tiktoken
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from tavily import TavilyClient
from colorama import init, Fore
# Initialize colorama
init(autoreset=True)
# GraphRAG related imports
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_communities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_report_embeddings
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.drift_search.drift_context import (
    DRIFTSearchContextBuilder,
)
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
# Athene-V2-Chat_exl2_2.25bpw，Rombos-Coder-V2.5-Qwen-32b-exl2_5.0bpw
LLM_MODEL = "Rombos-LLM-V2.5-Qwen-32b-4.5bpw-exl2"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up constants and configurations
INPUT_DIR = "E:\\graphrag_kb\\input\\artifacts"
LANCEDB_URI = "E:\\graphrag_kb\\output\\lancedb"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
FINAL_COMMUNITY_TABLE = "create_final_communities"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 10     # Community level, the higher the level, the more detailed community reports are used (but higher computational cost), default 2
PORT = 8013

# Global variables for storing search engines and question generators
local_search_engine = None
global_search_engine = None
drift_serch_engine = None
question_generator = None


# Data models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 0.7
    n: Optional[int] = 1
    stream: Optional[bool] = True
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 12_000
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


async def setup_llm_and_embedder():
    """
    Set up the language model (LLM) and embedding model
    When the knowledge graph is not working properly, retrain the small document generation graph with the gemini-1.5-flash-latest model
    """
    """
    # Get API key and base URL
    api_key = "xxx"
    api_key_embedding = "xxx"
    api_base = "https://ai.liaobots.work/v1"
    api_base_embedding = "https://ai.liaobots.work/v1"
    api_base = "http://localhost:11434/v1"
    # Get model name
    LLM_MODEL = "gpt-4o-mini"
    embedding_model = "text-embedding-ada-002"
    """
    logger.info("Setting up LLM and embedder")
    # ollama gets the API key and base URL
    api_key = "xxx"
    api_base = "http://127.0.0.1:5001/v1"
    global LLM_MODEL
    logger.info(Fore.CYAN + f"GRAPHRAG using model: {LLM_MODEL}")
    # Initialize ChatOpenAI instance
    llm = ChatOpenAI(
        api_key=api_key,
        api_base=api_base,
        model=LLM_MODEL,
        api_type=OpenaiApiType.OpenAI,
        max_retries=10,
        request_timeout=120  # Set the timeout to 120 seconds
    )

    # Initialize token encoder
    token_encoder = tiktoken.get_encoding("cl100k_base")

    # Initialize text embedding model
    # Openai online model
    """
    api_key="sk-9mxwRPHwHt8M1ct7CaCf041d6fC44e9587A041Ca3145E09e",
    api_base="https://apis.wumingai.com/v1",
    model=embedding_model,
    deployment_name=embedding_model,
    """
    # xinference local embedding model
    """
    api_key="xinference",
    api_base="http://127.0.0.1:9997/v1",
    model="bge-m3",
    deployment_name="bge-m3",
    """
    text_embedder = OpenAIEmbedding(
        # Local embedding model
        api_key="ollama",
        api_base="http://localhost:11434/v1",
        model="bge-m3:Q4",
        deployment_name="bge-m3:Q4",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    logger.info("LLM and embedder setup complete")
    return llm, token_encoder, text_embedder

def embed_text(column):
    text_embedder = OpenAIEmbedding(
        # Local embedding model
        api_key="ollama",
        api_base="http://localhost:11434/v1",
        model="bge-m3:Q4",
        deployment_name="bge-m3:Q4",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    return column.apply(lambda x: text_embedder.embed(x))
async def load_context():
    """
    Load context data, including entities, relationships, reports, text units, and covariates
    """
    logger.info("Loading context data")
    try:
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
        description_embedding_store.connect(db_uri=LANCEDB_URI)

        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        relationships = read_indexer_relationships(relationship_df)

        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        #reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL,content_embedding_col="full_content_embeddings")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL,content_embedding_col="full_content_embeddings")
        full_content_embedding_store = LanceDBVectorStore(collection_name="default-community-full_content")
        full_content_embedding_store.connect(db_uri=LANCEDB_URI)
        read_indexer_report_embeddings(reports, full_content_embedding_store)
        
        final_communities_df = pd.read_parquet(f"{INPUT_DIR}/{FINAL_COMMUNITY_TABLE}.parquet")
        communities = read_indexer_communities(final_communities_df,entity_df,report_df)


        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)
        covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
        claims = read_indexer_covariates(covariate_df)
        logger.info(f"Number of claims: {len(claims)}")
        covariates = {"claims": claims}

        logger.info("Context data loading complete")
        return entities, relationships, reports, communities,text_units, description_embedding_store, covariates
    except Exception as e:
        logger.error(f"Error loading context data: {str(e)}")
        raise


async def setup_search_engines(llm, token_encoder, text_embedder, entities, relationships, reports,communities, text_units,
                            description_embedding_store, covariates):
    """
    Set up local search engine and global search engine
    """
    logger.info("Setting up search engines")

    # Set up local search engine
    local_context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    local_llm_params = {
        "max_tokens": 12_000,
        "temperature": 0.3,
    }

    local_search_engine = LocalSearch(
        llm=llm,
        context_builder=local_context_builder,
        token_encoder=token_encoder,
        llm_params=local_llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )
    # Set up global search engine
    global_context_builder = GlobalCommunityContext(
        communities = communities ,
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    global_context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0.5,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 12_000,
        "temperature": 0.5,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 12_000,
        "temperature": 0.5,
    }

    global_search_engine = GlobalSearch(
        llm=llm,
        context_builder=global_context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=True,
        json_mode=True,
        context_builder_params=global_context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )
    drift_params = DRIFTSearchConfig()
    drift_params.temperature = 0.5
    drift_params.max_tokens = 12_000
    drift_context_builder = DRIFTSearchContextBuilder(
            chat_llm=llm,
            text_embedder=text_embedder,
            entities=entities,
            relationships=relationships,
            reports=reports,
            entity_text_embeddings=description_embedding_store,
            text_units=text_units,
            config = drift_params

        )
    drift_serch_engine = DRIFTSearch(
            llm=llm, context_builder=drift_context_builder, token_encoder=token_encoder
        )
    logger.info("Search engine setup complete")
    return local_search_engine, global_search_engine, drift_serch_engine, local_context_builder, local_llm_params, local_context_params


def format_response(response):
    """
    Format the response, adding appropriate line breaks and paragraph breaks.
    """
    modified_text = re.sub(r"`(.*?)`", r'```python\1```', response)

    return modified_text


async def tavily_search(prompt: str):
    """
    Use the Tavily API for search
    """
    try:
        client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
        resp = client.search(prompt, search_depth="advanced")

        # Convert the Tavily response to Markdown format
        markdown_response = "# Search Results\n\n"
        for result in resp.get('results', []):
            markdown_response += f"## [{result['title']}]({result['url']})\n\n"
            markdown_response += f"{result['content']}\n\n"

        return markdown_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tavily search error: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Execute on startup
    global local_search_engine, global_search_engine,drift_serch_engine, question_generator
    try:
        logger.info("Initializing search engines and question generators...")

        llm, token_encoder, text_embedder = await setup_llm_and_embedder()
        entities, relationships, reports,communities, text_units, description_embedding_store, covariates = await load_context()
        local_search_engine, global_search_engine,drift_serch_engine, local_context_builder, local_llm_params, local_context_params = await setup_search_engines(
            llm, token_encoder, text_embedder, entities, relationships, reports,communities, text_units,
            description_embedding_store, covariates
        )

        question_generator = LocalQuestionGen(
            llm=llm,
            context_builder=local_context_builder,
            token_encoder=token_encoder,
            llm_params=local_llm_params,
            context_builder_params=local_context_params,
        )
        logger.info("Initialization complete.")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

    yield
    # Execute on shutdown
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

# Add the following code to the chat_completions function

async def full_model_search(prompt: str):
    """
    Perform full model search, including local search, global search, drift search, and Tavily search
    """
    local_result = await local_search_engine.asearch(prompt)
    global_result = await global_search_engine.asearch(prompt)
    drift_result = await drift_serch_engine.asearch(prompt)
    tavily_result = await tavily_search(prompt)

    # Format the results
    formatted_result = "# 🔥🔥🔥 Comprehensive Search Results\n\n"

    formatted_result += "## 🔥🔥🔥 Local Search Results\n"
    formatted_result += local_result.response + "\n\n"

    formatted_result += "## 🔥🔥🔥 Global Search Results\n"
    formatted_result += global_result.response + "\n\n"

    formatted_result += "## 🔥🔥🔥 Drift Search Results\n"
    formatted_result += drift_result.response + "\n\n"

    formatted_result += "## 🔥🔥🔥 Tavily Search Results\n"
    formatted_result += tavily_result + "\n\n"

    return formatted_result

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not local_search_engine or not global_search_engine or not drift_serch_engine:
        logger.error("Search engines not initialized")
        raise HTTPException(status_code=500, detail="Search engines not initialized")

    prompt = request.messages[-1].content
    logger.info(Fore.CYAN + f"Received model request content: {prompt}")
    # Use different search methods based on the model
    if request.model == "graphrag-global-search:latest":
        result = await global_search_engine.asearch(prompt)
        formatted_response = result.response
    elif request.model == "graphrag-drift-search:latest":
        result = await drift_serch_engine.asearch(prompt)
        formatted_response = result.response
        formatted_response:str = formatted_response["nodes"][0]["answer"]
        formatted_response = formatted_response.replace(" n n","\n")
    elif request.model == "tavily-search:latest":
        result = await tavily_search(prompt)
        formatted_response = result
    elif request.model == "full-model:latest":
        formatted_response = await full_model_search(prompt)
    else:  # Default to using local search
        result = await local_search_engine.asearch(prompt)
        # Format the response
        #formatted_response = format_response(result.response)
        formatted_response = result.response

    logger.info(Fore.CYAN + f"Knowledge graph search result: {formatted_response}")
    # Stream and non-stream response handling remains unchanged
    if request.stream:
        async def generate_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            lines = formatted_response.split('\n')
            for i, line in enumerate(lines):
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            #"delta": {"content": line + '\n'} if i > 0 else {"role": "assistant", "content": ""},
                            "delta": {"content": line + '\n'}, # if i > 0 else {"role": "assistant", "content": ""},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)

            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        response = ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=formatted_response),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(formatted_response.split()),
                total_tokens=len(prompt.split()) + len(formatted_response.split())
            )
        )
    logger.info(f"Sending response: {response}")
    return JSONResponse(content=response.dict())

@app.get("/v1/models")
async def list_models():
    """
    Return the list of available models
    """
    logger.info("Received model list request")
    current_time = int(time.time())
    models = [
        {"id": "graphrag-local-search:latest", "object": "model", "created": current_time - 100000, "owned_by": "graphrag"},
        {"id": "graphrag-global-search:latest", "object": "model", "created": current_time - 95000, "owned_by": "graphrag"},
        {"id": "graphrag-drift-search:latest", "object": "model", "created": current_time - 90000, "owned_by": "graphrag"},
        # {"id": "graphrag-question-generator:latest", "object": "model", "created": current_time - 90000, "owned_by": "graphrag"},
        # {"id": "gpt-3.5-turbo:latest", "object": "model", "created": current_time - 8_0000, "owned_by": "openai"},
        # {"id": "text-embedding-3-small:latest", "object": "model", "created": current_time - 70000, "owned_by": "openai"},
        #{"id": "tavily-search:latest", "object": "model", "created": current_time - 85000, "owned_by": "tavily"},
        # {"id": "full-model:latest", "object": "model", "created": current_time - 8_0000, "owned_by": "combined"}

    ]

    response = {
        "object": "list",
        "data": models
    }

    logger.info(f"Sending model list: {response}")
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

