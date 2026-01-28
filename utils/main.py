"""
GraphRAG FastAPI 服务器

该脚本实现了一个 FastAPI 服务器，提供知识图谱问答接口。
支持本地搜索、全局搜索和综合搜索三种模式。

日期: 2026-01-27
作者: LiuJunDa
"""

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
import uvicorn

# GraphRAG 相关导入
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 设置常量和配置。INPUT_DIR 根据自己建立的 graphrag 文件夹路径进行修改
INPUT_DIR = "D:/PythonWork/RAG/graph_test/ragtest/inputs/artifacts"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2
PORT = 8012

# 全局变量，用于存储搜索引擎和问题生成器
local_search_engine = None
global_search_engine = None
question_generator = None


# 定义 Message 类型
class Message(BaseModel):
    role: str
    content: str


# 定义 ChatCompletionRequest 类
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


# 定义 ChatCompletionResponseChoice 类
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


# 定义 Usage 类
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# 定义 ChatCompletionResponse 类
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
    设置语言模型（LLM）、token 编码器（TokenEncoder）和文本嵌入向量生成器（TextEmbedder）
    """
    logger.info("正在设置 LLM 和嵌入器")
    # 实例化一个 ChatOpenAI 客户端对象
    llm = ChatOpenAI(
        # # 调用 gpt
        # api_base="https://api.wlai.vip/v1",  # 请求的 API 服务地址
        # api_key="sk-4P8HC2GD6heTwx0l8dD83f13F1014e039eC4Ac6d47877dCb",  # API Key
        # model="gpt-4o-mini",  # 本次使用的模型
        # api_type=OpenaiApiType.OpenAI,

        # # 调用其他模型  通过 oneAPI
        # api_base="http://139.224.72.218:3000/v1",  # 请求的 API 服务地址
        # api_key="sk-KtEtYw4jOGtSpr4n2e06Ee978690452183Be8a1fF75cA8C5",  # API Key
        # model="qwen-plus",  # 本次使用的模型
        # api_type=OpenaiApiType.OpenAI,

        # 调用本地大模型  通过 Ollama
        api_base="http://127.0.0.1:3000/v1",  # 请求的 API 服务地址
        api_key="sk-PrMSNlbfrlQH5lm1A6Ca01Ba63194892B4001d210fDf9cA9",  # API Key
        model="qwen-plus",  # 本次使用的模型
        api_type=OpenaiApiType.OpenAI,
    )

    # 初始化 token 编码器
    token_encoder = tiktoken.get_encoding("cl100k_base")

    # 实例化 OpenAIEmbedding 处理模型
    text_embedder = OpenAIEmbedding(
        # # 调用 gpt
        # api_base="https://api.wlai.vip/v1",  # 请求的 API 服务地址
        # api_key="sk-Soz7kmey8JKidej0AeD416B87d2547E1861d29F4F3E7A75e",  # API Key
        # model="text-embedding-3-small",
        # deployment_name="text-embedding-3-small",
        # api_type=OpenaiApiType.OpenAI,
        # max_retries=20,

        # # 调用其他模型  通过 oneAPI
        # api_base="http://139.224.72.218:3000/v1",  # 请求的 API 服务地址
        # api_key="sk-KtEtYw4jOGtSpr4n2e06Ee978690452183Be8a1fF75cA8C5",  # API Key
        # model="text-embedding-v1",
        # deployment_name="text-embedding-v1",
        # api_type=OpenaiApiType.OpenAI,
        # max_retries=20,

        # 调用本地大模型  通过 Ollama
        api_base="http://127.0.0.1:3000/v1",  # 请求的 API 服务地址
        api_key="sk-PrMSNlbfrlQH5lm1A6Ca01Ba63194892B4001d210fDf9cA9",  # API Key
        model="text-embedding-v1",
        deployment_name="text-embedding-v1",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
        
    )

    logger.info("LLM 和嵌入器设置完成")
    return llm, token_encoder, text_embedder


async def load_context():
    """
    加载上下文数据，包括实体、关系、报告、文本单元和协变量
    """
    logger.info("正在加载上下文数据")
    try:
        # 使用 pandas 库从指定的路径读取实体数据表 ENTITY_TABLE，文件格式为 Parquet，并将其加载为 DataFrame
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        # 读取实体嵌入向量数据表 ENTITY_EMBEDDING_TABLE，并将其加载为 DataFrame
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
        # 将 entity_df 和 entity_embedding_df 传入，并基于 COMMUNITY_LEVEL（社区级别）处理这些数据
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
        # 创建一个 LanceDBVectorStore 的实例，用于存储实体的描述嵌入向量
        description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
        # 通过调用 connect 方法，连接到指定的 LanceDB 数据库
        description_embedding_store.connect(db_uri=LANCEDB_URI)
        # 将已处理的实体数据存储到 description_embedding_store 中，用于语义搜索或其他用途
        store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)
        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        relationships = read_indexer_relationships(relationship_df)
        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)
        covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
        claims = read_indexer_covariates(covariate_df)
        logger.info(f"声明记录数: {len(claims)}")
        covariates = {"claims": claims}
        logger.info("上下文数据加载完成")
        return entities, relationships, reports, text_units, description_embedding_store, covariates
    except Exception as e:
        logger.error(f"加载上下文数据时出错: {str(e)}")
        raise

async def setup_search_engines(llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
                               description_embedding_store, covariates):
    """
    设置本地和全局搜索引擎、上下文构建器（ContextBuilder）、以及相关参数
    """
    logger.info("正在设置搜索引擎")
    # 设置本地搜索引擎
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
        # "max_tokens": 12_000,
        "max_tokens": 4096,
    }

    local_llm_params = {
        # "max_tokens": 2_000,
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    local_search_engine = LocalSearch(
        llm=llm,
        context_builder=local_context_builder,
        token_encoder=token_encoder,
        llm_params=local_llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )

    # 设置全局搜索引擎
    global_context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    global_context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        # "max_tokens": 12_000,
        "max_tokens": 4096,
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
        llm=llm,
        context_builder=global_context_builder,
        token_encoder=token_encoder,
        # max_data_tokens=12_000,
        max_data_tokens=4096,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=global_context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )

    logger.info("搜索引擎设置完成")
    return local_search_engine, global_search_engine, local_context_builder, local_llm_params, local_context_params


def format_response(response):
    """
    格式化响应，对输入的文本进行段落分隔、添加适当的换行符，以及在代码块中增加标记
    
    具体步骤:
        1. 使用正则表达式按照两个或更多的连续换行符分割文本为多个段落
        2. 检查每个段落中是否包含代码块标记
        3. 若包含代码块，则将代码块部分用换行符和 ``` 包围
        4. 若不包含代码块，则将句号后面的空格替换为换行符
        5. 将所有格式化后的段落用两个换行符连接起来
    """
    # 使用正则表达式 \n{2, } 将输入的 response 按照两个或更多的连续换行符进行分割
    paragraphs = re.split(r'\n{2,}', response)
    # 空列表，用于存储格式化后的段落
    formatted_paragraphs = []
    # 遍历每个段落进行处理
    for para in paragraphs:
        # 检查段落中是否包含代码块标记
        if '```' in para:
            # 将段落按照``` 分割成多个部分，代码块和普通文本交替出现
            parts = para.split('```')
            for i, part in enumerate(parts):
                # 检查当前部分的索引是否为奇数，奇数部分代表代码块
                if i % 2 == 1:  # 这是代码块
                    # 将代码块部分用换行符和``` 包围，并去除多余的空白字符
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            # 将分割后的部分重新组合成一个字符串
            para = ''.join(parts)
        else:
            # 否则，将句子中的句号后面的空格替换为换行符，以便句子之间有明确的分隔
            para = para.replace('. ', '.\n')
        # 将格式化后的段落添加到 formatted_paragraphs 列表
        # strip() 方法用于移除字符串开头和结尾的空白字符（包括空格、制表符 \t、换行符 \n 等）
        formatted_paragraphs.append(para.strip())
    # 将所有格式化后的段落用两个换行符连接起来，以形成一个具有清晰段落分隔的文本
    return '\n\n'.join(formatted_paragraphs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    定义了一个异步函数，用于管理应用的生命周期
    
    函数在应用启动时执行初始化操作，如设置搜索引擎、加载上下文数据、以及初始化问题生成器
    函数在应用关闭时执行清理操作
    
    @asynccontextmanager 装饰器用于创建一个异步上下文管理器
    """
    # 启动时执行
    # 申明引用全局变量，在函数中被初始化，并在整个应用中使用
    global local_search_engine, global_search_engine, question_generator
    try:
        logger.info("正在初始化搜索引擎和问题生成器...")
        # 调用 setup_llm_and_embedder() 函数以设置语言模型（LLM）、token 编码器（TokenEncoder）和文本嵌入向量生成器（TextEmbedder）
        # await 关键字表示此调用是异步的，函数将在这个操作完成后继续执行
        llm, token_encoder, text_embedder = await setup_llm_and_embedder()
        # 调用 load_context() 函数加载实体、关系、报告、文本单元、描述嵌入存储和协变量等数据
        entities, relationships, reports, text_units, description_embedding_store, covariates = await load_context()
        # 调用 setup_search_engines() 函数设置本地和全局搜索引擎、上下文构建器（ContextBuilder）、以及相关参数
        local_search_engine, global_search_engine, local_context_builder, local_llm_params, local_context_params = await setup_search_engines(
            llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
            description_embedding_store, covariates
        )
        # 使用 LocalQuestionGen 类创建一个本地问题生成器 question_generator
        question_generator = LocalQuestionGen(
            llm=llm,
            context_builder=local_context_builder,
            token_encoder=token_encoder,
            llm_params=local_llm_params,
            context_builder_params=local_context_params,
        )
        logger.info("初始化完成")
    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        # raise 关键字重新抛出异常，以确保程序不会在错误状态下继续运行
        raise
    # yield 关键字将控制权交还给 FastAPI 框架，使应用开始运行
    # 分隔了启动和关闭的逻辑。在 yield 之前的代码在应用启动时运行，yield 之后的代码在应用关闭时运行
    yield

    # 关闭时执行
    logger.info("正在关闭...")

# lifespan 参数用于在应用程序生命周期的开始和结束时执行一些初始化或清理工作
app = FastAPI(lifespan=lifespan)


async def full_model_search(prompt: str):
    """
    执行全模型搜索，包括本地检索、全局检索
    """
    local_result = await local_search_engine.asearch(prompt)
    global_result = await global_search_engine.asearch(prompt)
    # 格式化结果
    formatted_result = "#综合搜索结果:\n\n"
    formatted_result += "##本地检索结果:\n"
    formatted_result += format_response(local_result.response) + "\n\n"
    formatted_result += "##全局检索结果:\n"
    formatted_result += format_response(global_result.response) + "\n\n"
    return formatted_result


# POST 请求接口，与大模型进行知识问答
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not local_search_engine or not global_search_engine:
        logger.error("搜索引擎未初始化")
        raise HTTPException(status_code=500, detail="搜索引擎未初始化")

    try:
        logger.info(f"收到聊天完成请求: {request}")
        prompt = request.messages[-1].content
        logger.info(f"处理提示: {prompt}")

        # 根据模型选择使用不同的搜索方法
        if request.model == "graphrag-global-search:latest":
            result = await global_search_engine.asearch(prompt)
            formatted_response = format_response(result.response)
        elif request.model == "full-model:latest":
            formatted_response = await full_model_search(prompt)
        elif request.model == "graphrag-local-search:latest":  # 默认使用本地搜索
            result = await local_search_engine.asearch(prompt)
            formatted_response = format_response(result.response)

        logger.info(f"格式化的搜索结果:\n {formatted_response}")

        # 流式响应和非流式响应的处理
        if request.stream:
            # 定义一个异步生成器函数，用于生成流式数据
            async def generate_stream():
                # 为每个流式数据片段生成一个唯一的 chunk_id
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                # 将格式化后的响应按行分割
                lines = formatted_response.split('\n')
                # 遍历每一行，并构建响应片段
                for i, line in enumerate(lines):
                    # 创建一个字典，表示流式数据的一个片段
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": line + '\n'},  # if i > 0 else {"role": "assistant", "content": ""},
                                "finish_reason": None
                            }
                        ]
                    }
                    # 将片段转换为 JSON 格式并生成
                    yield f"data: {json.dumps(chunk)}\n"
                    # 每次生成数据后，异步等待 0.5 秒
                    await asyncio.sleep(0.5)
                # 生成最后一个片段，表示流式响应的结束
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
                yield f"data: {json.dumps(final_chunk)}\n"
                yield "data: [DONE]\n"

            # 返回 StreamingResponse 对象，流式传输数据，media_type 设置为 text/event-stream 以符合 SSE(Server-Sent Events) 格式
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        # 非流式响应处理
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
                # 使用情况
                usage=Usage(
                    # 提示文本的 tokens 数量
                    prompt_tokens=len(prompt.split()),
                    # 完成文本的 tokens 数量
                    completion_tokens=len(formatted_response.split()),
                    # 总 tokens 数量
                    total_tokens=len(prompt.split()) + len(formatted_response.split())
                )
            )
            logger.info(f"发送响应: \n\n{response}")
            # 返回 JSONResponse 对象，其中 content 是将 response 对象转换为字典的结果
            return JSONResponse(content=response.dict())

    except Exception as e:
        logger.error(f"处理聊天完成时出错:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# GET 请求接口，获取可用模型列表
@app.get("/v1/models")
async def list_models():
    logger.info("收到模型列表请求")
    current_time = int(time.time())
    models = [
        {"id": "graphrag-local-search:latest", "object": "model", "created": current_time - 100000, "owned_by": "graphrag"},
        {"id": "graphrag-global-search:latest", "object": "model", "created": current_time - 95000, "owned_by": "graphrag"},
        {"id": "full-model:latest", "object": "model", "created": current_time - 80000, "owned_by": "combined"}
    ]

    response = {
        "object": "list",
        "data": models
    }

    logger.info(f"发送模型列表: {response}")
    return JSONResponse(content=response)



if __name__ == "__main__":
    logger.info(f"在端口 {PORT} 上启动服务器")
    # uvicorn 是一个用于运行 ASGI 应用的轻量级、超快速的 ASGI 服务器实现
    # 用于部署基于 FastAPI 框架的异步 Python Web 应用程序
    uvicorn.run(app, host="0.0.0.0", port=PORT)

