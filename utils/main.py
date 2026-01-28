"""
GraphRAG FastAPI 服务器 (适配 GraphRAG 2.7.0)

该脚本实现了一个 FastAPI 服务器，提供知识图谱问答接口。
支持本地搜索、全局搜索和综合搜索三种模式。

日期: 2026-01-28
作者: LiuJunDa
版本: 2.0.1 (修复版 - 适配 GraphRAG 2.7.0)
"""

import os
import asyncio
import time
import uuid
import json
import re
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from contextlib import asynccontextmanager
import uvicorn

# GraphRAG 2.7.0 导入
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.models.vector_store_config import VectorStoreConfig
from graphrag.language_model.manager import ModelManager
from graphrag.tokenizer.get_tokenizer import get_tokenizer
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_communities,
)
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== 配置区域 ====================
# 请根据你的实际路径和 API 设置修改以下配置

# GraphRAG 项目根目录
GRAPHRAG_ROOT = "/home/sunlight/Projects/graphrag-oneapi-exp/output"

# 索引输出目录（parquet 文件所在位置）
# 注意：GraphRAG 2.x 版本输出目录是 output/
# 请检查你的实际目录结构
INPUT_DIR = f"{GRAPHRAG_ROOT}"

# LanceDB 向量数据库路径
LANCEDB_URI = f"{GRAPHRAG_ROOT}/lancedb"

# 数据表名称
# GraphRAG 2.x 新版使用简化名称，旧版使用 create_final_* 前缀
# 请根据你的 parquet 文件名调整
USE_NEW_TABLE_NAMES = True  # 设置为 True 如果你的文件名是简化格式

if USE_NEW_TABLE_NAMES:
    # 新版文件名格式
    COMMUNITY_TABLE = "communities"
    COMMUNITY_REPORT_TABLE = "community_reports"
    ENTITY_TABLE = "entities"
    RELATIONSHIP_TABLE = "relationships"
    COVARIATE_TABLE = "covariates"
    TEXT_UNIT_TABLE = "text_units"
else:
    # 旧版文件名格式 (create_final_* 前缀)
    COMMUNITY_TABLE = "create_final_communities"  # 可能不存在
    COMMUNITY_REPORT_TABLE = "community_reports"
    ENTITY_TABLE = "entities"
    RELATIONSHIP_TABLE = "relationships"
    COVARIATE_TABLE = "covariates"
    TEXT_UNIT_TABLE = "text_units"

# 社区层级
COMMUNITY_LEVEL = 2

# API 服务端口
PORT = 8012

# LLM 配置
LLM_API_BASE = "http://127.0.0.1:3000/v1"
LLM_API_KEY = "sk-wZqb0cY5ONBebr2I9e32A07b22F24fFdA03eB53dC563223d"
LLM_MODEL = "qwen-plus"

# Embedding 配置
EMBEDDING_API_BASE = "http://127.0.0.1:3000/v1"
EMBEDDING_API_KEY = "sk-wZqb0cY5ONBebr2I9e32A07b22F24fFdA03eB53dC563223d"
EMBEDDING_MODEL = "text-embedding-v1"

# ==================== 配置区域结束 ====================


# 全局变量
local_search_engine = None
global_search_engine = None
question_generator = None


# Pydantic 模型定义
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
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


def find_parquet_file(base_dir: str, possible_names: List[str]) -> Optional[str]:
    """
    在目录中查找 parquet 文件，支持多种可能的文件名
    """
    for name in possible_names:
        path = f"{base_dir}/{name}.parquet"
        if os.path.exists(path):
            logger.info(f"找到文件: {path}")
            return path
    return None


async def setup_llm_and_embedder():
    """
    设置语言模型（LLM）和文本嵌入模型（Embedder）
    """
    logger.info("正在设置 LLM 和嵌入器 (GraphRAG 2.7.0 API)")
    
    # 配置 Chat 模型
    chat_config = LanguageModelConfig(
        api_key=LLM_API_KEY,
        api_base=LLM_API_BASE,
        type=ModelType.Chat,
        model=LLM_MODEL,
        model_provider="openai",
        max_retries=20,
    )
    
    # 创建 Chat 模型实例
    chat_model = ModelManager().get_or_create_chat_model(
        name="graphrag_chat",
        model_type=ModelType.Chat,
        config=chat_config,
    )
    
    # 获取 tokenizer
    tokenizer = get_tokenizer(chat_config)
    
    # 配置 Embedding 模型
    embedding_config = LanguageModelConfig(
        api_key=EMBEDDING_API_KEY,
        api_base=EMBEDDING_API_BASE,
        type=ModelType.Embedding,
        model=EMBEDDING_MODEL,
        model_provider="openai",
        max_retries=20,
    )
    
    # 创建 Embedding 模型实例
    text_embedder = ModelManager().get_or_create_embedding_model(
        name="graphrag_embedding",
        model_type=ModelType.Embedding,
        config=embedding_config,
    )

    logger.info("LLM 和嵌入器设置完成")
    return chat_model, tokenizer, text_embedder


def load_context():
    """
    加载上下文数据
    """
    logger.info(f"正在从 {INPUT_DIR} 加载上下文数据")
    
    try:
        # 读取实体数据
        entity_path = find_parquet_file(INPUT_DIR, ["entities", "create_final_entities", "create_final_nodes"])
        if not entity_path:
            raise FileNotFoundError(f"未找到实体文件，请检查 {INPUT_DIR} 目录")
        entity_df = pd.read_parquet(entity_path)
        logger.info(f"实体记录数: {len(entity_df)}")
        
        # 读取社区数据（可能不存在于所有版本）
        community_path = find_parquet_file(INPUT_DIR, ["communities", "create_final_communities"])
        if community_path:
            community_df = pd.read_parquet(community_path)
            logger.info(f"社区记录数: {len(community_df)}")
        else:
            logger.warning("未找到社区文件，将使用实体数据构建")
            community_df = None
        
        # 读取社区报告（提前读取，用于后续 communities 解析）
        report_path = find_parquet_file(INPUT_DIR, ["community_reports", "create_final_community_reports"])
        report_df = None
        if report_path:
            report_df = pd.read_parquet(report_path)
            logger.info(f"社区报告记录数: {len(report_df)}")
        else:
            logger.warning("未找到社区报告文件")
        
        # 解析实体
        if community_df is not None:
            entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)
        else:
            entity_embedding_path = find_parquet_file(INPUT_DIR, ["create_final_entities", "entities"])
            if entity_embedding_path and entity_embedding_path != entity_path:
                entity_embedding_df = pd.read_parquet(entity_embedding_path)
                entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
            else:
                entities = read_indexer_entities(entity_df, entity_df, COMMUNITY_LEVEL)
        logger.info(f"解析后实体数: {len(entities)}")
        
        # 读取社区信息（用于 GlobalSearch）- 使用 report_df
        communities = None
        if community_df is not None and report_df is not None:
            try:
                communities = read_indexer_communities(community_df, report_df)
                logger.info("社区信息解析完成")
            except Exception as e:
                logger.warning(f"读取社区信息失败: {e}")
        
        # 连接 LanceDB 向量存储
        logger.info(f"连接 LanceDB: {LANCEDB_URI}")
        vector_store_schema = VectorStoreSchemaConfig(
            index_name="default-entity-description"
        )
        description_embedding_store = LanceDBVectorStore(
            vector_store_schema_config=vector_store_schema
        )
        description_embedding_store.connect(db_uri=LANCEDB_URI)
        
        # 读取关系数据
        relationship_path = find_parquet_file(INPUT_DIR, ["relationships", "create_final_relationships"])
        if relationship_path:
            relationship_df = pd.read_parquet(relationship_path)
            relationships = read_indexer_relationships(relationship_df)
            logger.info(f"关系记录数: {len(relationships)}")
        else:
            relationships = []
            logger.warning("未找到关系文件")
        
        # 解析社区报告
        if report_df is not None:
            if community_df is not None:
                reports = read_indexer_reports(report_df, community_df, COMMUNITY_LEVEL)
            else:
                reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
            logger.info(f"报告记录数: {len(reports)}")
        else:
            reports = []
        
        # 读取文本单元
        text_unit_path = find_parquet_file(INPUT_DIR, ["text_units", "create_final_text_units"])
        if text_unit_path:
            text_unit_df = pd.read_parquet(text_unit_path)
            text_units = read_indexer_text_units(text_unit_df)
            logger.info(f"文本单元数: {len(text_units)}")
        else:
            text_units = []
            logger.warning("未找到文本单元文件")
        
        # 读取协变量（可选）
        covariates = None
        covariate_path = find_parquet_file(INPUT_DIR, ["covariates", "create_final_covariates"])
        if covariate_path:
            try:
                covariate_df = pd.read_parquet(covariate_path)
                claims = read_indexer_covariates(covariate_df)
                logger.info(f"声明记录数: {len(claims)}")
                covariates = {"claims": claims}
            except Exception as e:
                logger.warning(f"读取协变量失败: {e}")
        
        logger.info("上下文数据加载完成")
        return entities, relationships, reports, text_units, description_embedding_store, covariates, communities
    
    except Exception as e:
        logger.error(f"加载上下文数据时出错: {str(e)}")
        raise

async def setup_search_engines(
    chat_model, 
    tokenizer, 
    text_embedder, 
    entities, 
    relationships, 
    reports, 
    text_units,
    description_embedding_store, 
    covariates,
    communities
):
    """
    设置本地和全局搜索引擎
    """
    logger.info("正在设置搜索引擎")
    
    # ==================== 本地搜索引擎设置 ====================
    local_context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        tokenizer=tokenizer,
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

    local_model_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }

    local_search_engine = LocalSearch(
        model=chat_model,
        context_builder=local_context_builder,
        tokenizer=tokenizer,
        model_params=local_model_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )

    # ==================== 全局搜索引擎设置 ====================
    global_context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,
        tokenizer=tokenizer,
    )

    global_context_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    global_search_engine = GlobalSearch(
        model=chat_model,
        context_builder=global_context_builder,
        tokenizer=tokenizer,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=global_context_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )

    logger.info("搜索引擎设置完成")
    return local_search_engine, global_search_engine, local_context_builder, local_model_params, local_context_params


def format_response(response):
    """
    格式化响应
    """
    paragraphs = re.split(r'\n{2,}', response)
    formatted_paragraphs = []
    for para in paragraphs:
        if '```' in para:
            parts = para.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = ''.join(parts)
        else:
            para = para.replace('. ', '.\n')
        formatted_paragraphs.append(para.strip())
    return '\n\n'.join(formatted_paragraphs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    管理应用的生命周期
    """
    global local_search_engine, global_search_engine, question_generator
    
    try:
        logger.info("正在初始化搜索引擎 (GraphRAG 2.7.0)...")
        
        # 设置 LLM 和 Embedder
        chat_model, tokenizer, text_embedder = await setup_llm_and_embedder()
        
        # 加载上下文数据
        entities, relationships, reports, text_units, description_embedding_store, covariates, communities = load_context()
        
        # 设置搜索引擎
        local_search_engine, global_search_engine, local_context_builder, local_model_params, local_context_params = await setup_search_engines(
            chat_model, tokenizer, text_embedder, entities, relationships, reports, text_units,
            description_embedding_store, covariates, communities
        )
        
        # 设置问题生成器
        question_generator = LocalQuestionGen(
            model=chat_model,
            context_builder=local_context_builder,
            tokenizer=tokenizer,
            model_params=local_model_params,
            context_builder_params=local_context_params,
        )
        
        logger.info("✅ 初始化完成")
    except Exception as e:
        logger.error(f"❌ 初始化过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    logger.info("正在关闭...")


app = FastAPI(
    title="GraphRAG API Server",
    description="基于 Microsoft GraphRAG 2.7.0 的知识图谱问答服务",
    version="2.0.1",
    lifespan=lifespan
)

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def full_model_search(prompt: str):
    """
    执行全模型搜索
    """
    if not local_search_engine or not global_search_engine:
        raise ValueError("搜索引擎未初始化")
    
    local_result = await local_search_engine.search(prompt)
    global_result = await global_search_engine.search(prompt)
    
    formatted_result = "# 综合搜索结果:\n\n"
    formatted_result += "## 本地检索结果:\n"
    formatted_result += format_response(local_result.response) + "\n\n"
    formatted_result += "## 全局检索结果:\n"
    formatted_result += format_response(global_result.response) + "\n\n"
    return formatted_result


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 兼容的聊天完成接口
    """
    if not local_search_engine or not global_search_engine:
        logger.error("搜索引擎未初始化")
        raise HTTPException(status_code=500, detail="搜索引擎未初始化")

    try:
        logger.info(f"收到请求，模型: {request.model}")
        prompt = request.messages[-1].content
        logger.info(f"处理提示: {prompt[:100]}...")

        # 根据模型选择搜索方法
        if request.model == "graphrag-global-search:latest":
            result = await global_search_engine.search(prompt)
            formatted_response = format_response(result.response)
        elif request.model == "full-model:latest":
            formatted_response = await full_model_search(prompt)
        else:  # 默认本地搜索
            result = await local_search_engine.search(prompt)
            formatted_response = format_response(result.response)

        logger.info(f"搜索完成，响应长度: {len(formatted_response)}")

        # 流式响应
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
                                "delta": {"content": line + '\n'},
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
        
        # 非流式响应
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
            return JSONResponse(content=response.model_dump())

    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """
    获取可用模型列表
    """
    current_time = int(time.time())
    models = [
        {"id": "graphrag-local-search:latest", "object": "model", "created": current_time - 100000, "owned_by": "graphrag"},
        {"id": "graphrag-global-search:latest", "object": "model", "created": current_time - 95000, "owned_by": "graphrag"},
        {"id": "full-model:latest", "object": "model", "created": current_time - 80000, "owned_by": "combined"}
    ]
    return JSONResponse(content={"object": "list", "data": models})


@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {
        "status": "healthy",
        "version": "2.0.1",
        "graphrag_version": "2.7.0",
        "local_search_ready": local_search_engine is not None,
        "global_search_ready": global_search_engine is not None,
    }


@app.get("/")
async def root():
    """
    根路径
    """
    return {
        "message": "GraphRAG API Server",
        "version": "2.0.1",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    logger.info(f"在端口 {PORT} 上启动 GraphRAG API 服务器")
    uvicorn.run(app, host="0.0.0.0", port=PORT)