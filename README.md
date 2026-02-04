# GraphRAG OneAPI 实验项目

基于微软 [GraphRAG](https://github.com/microsoft/graphrag) 2.7.0 的知识图谱问答系统，支持通过 OneAPI 接入多种大语言模型。

## 📋 项目简介

本项目实现了一个完整的 GraphRAG 知识图谱问答系统，主要功能包括：

- **知识图谱构建**：从文本数据中自动提取实体和关系，构建知识图谱
- **多模式搜索**：支持本地搜索（Local Search）、全局搜索（Global Search）和综合搜索
- **3D 可视化**：将知识图谱进行三维交互式可视化展示
- **Neo4j 集成**：支持将知识图谱导入 Neo4j 图数据库
- **网页爬虫**：内置爬虫工具，可爬取网页内容并转换为 Markdown 格式
- **API 服务**：提供 OpenAI 兼容的 RESTful API 接口

## 🏗️ 项目结构

```
graphrag-oneapi-exp/
├── back/                    # 后端服务
│   └── main.py             # FastAPI 服务器主程序
├── cache/                   # GraphRAG 缓存目录
├── infra/                   # 基础设施配置
│   └── neo4j/              # Neo4j Docker 配置
│       └── docker-compose.yml
├── input/                   # 输入数据目录
│   └── *.txt               # 原始文本文件
├── output/                  # GraphRAG 输出目录
│   └── lancedb/            # LanceDB 向量数据库
├── prompts/                 # 提示词模板
│   ├── extract_graph.txt           # 实体关系提取提示词
│   ├── summarize_descriptions.txt  # 描述总结提示词
│   ├── extract_claims.txt          # 声明提取提示词
│   ├── local_search_system_prompt.txt
│   ├── global_search_*.txt
│   └── ...
├── reports/                 # 报告输出目录
├── utils/                   # 工具脚本
│   ├── main.py             # FastAPI 服务器 (GraphRAG 2.7.0 适配版)
│   ├── graphrag3dknowledge.py  # 3D 知识图谱可视化
│   ├── neo4jTest.py        # Neo4j 数据导入脚本
│   ├── spider.py           # 网页爬虫
│   └── apiTest.py          # API 测试脚本
├── .env                     # 环境变量配置
├── settings.yaml            # GraphRAG 配置文件
└── requirements.txt         # Python 依赖
```

## 🚀 快速开始

### 1. 环境要求

- Python >= 3.10, < 3.13
- Conda（推荐）
- Docker & Docker Compose（如需使用 Neo4j）

### 2. 安装依赖

```bash
# 创建并激活 conda 环境
conda create -n graphrag-oneapi python=3.11
conda activate graphrag-oneapi

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 或编辑 `.env` 文件：

```env
# API 配置
GRAPHRAG_API_BASE=http://127.0.0.1:3000/v1
GRAPHRAG_CHAT_API_KEY=your-api-key
GRAPHRAG_CHAT_MODEL=qwen-plus
GRAPHRAG_EMBEDDING_API_KEY=your-api-key
GRAPHRAG_EMBEDDING_MODEL=text-embedding-v3

# 提示词文件路径
GRAPHRAG_ENTITY_EXTRACTION_PROMPT_FILE=prompts/extract_graph.txt
GRAPHRAG_SUMMARIZE_DESCRIPTIONS_PROMPT_FILE=prompts/summarize_descriptions.txt
GRAPHRAG_CLAIM_EXTRACTION_PROMPT_FILE=prompts/extract_claims.txt

# 数据目录
GRAPHRAG_INPUT_DIR=input
GRAPHRAG_CACHE_DIR=cache
GRAPHRAG_STORAGE_DIR=output
GRAPHRAG_REPORTING_DIR=reports
```

### 4. 构建知识图谱

将文本文件放入 `input/` 目录，然后运行：

```bash
# 初始化 GraphRAG 项目（首次运行）
graphrag init --root .

# 执行索引构建
graphrag index --root .
```

### 5. 启动 API 服务

```bash
# 使用 utils/main.py（推荐，适配 GraphRAG 2.7.0）
python utils/main.py

# 或使用 back/main.py
python back/main.py
```

服务启动后，API 将在 `http://localhost:8012` 运行。

## 📖 功能说明

### API 接口

服务提供 OpenAI 兼容的 API 接口：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 获取可用模型列表 |
| `/v1/chat/completions` | POST | 聊天补全（支持流式） |

**可用模型：**
- `graphrag-local-search` - 本地搜索模式
- `graphrag-global-search` - 全局搜索模式
- `graphrag-mix-search` - 综合搜索模式

**请求示例：**

```bash
curl -X POST http://localhost:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "graphrag-local-search",
    "messages": [{"role": "user", "content": "你的问题"}],
    "stream": false
  }'
```

### 3D 知识图谱可视化

```bash
python utils/graphrag3dknowledge.py --help

# 示例：可视化知识图谱
python utils/graphrag3dknowledge.py --input output --port 8080
```

### Neo4j 数据导入

首先启动 Neo4j 数据库：

```bash
cd infra/neo4j
docker-compose up -d
```

然后导入数据：

```bash
python utils/neo4jTest.py --folder output
```

Neo4j 浏览器访问地址：`http://localhost:17474`

### 网页爬虫

```bash
python utils/spider.py --help

# 示例：爬取网页并转换为 Markdown
python utils/spider.py --url https://example.com --output ./crawled_data --max-pages 50
```

### API 测试

```bash
python utils/apiTest.py
```

## ⚙️ 配置说明

### settings.yaml

主要配置项：

```yaml
models:
  default_chat_model:
    model: qwen-plus              # 聊天模型
    api_base: ${GRAPHRAG_API_BASE}
    concurrent_requests: 25       # 并发请求数
    
  default_embedding_model:
    model: text-embedding-v3      # 嵌入模型
    
chunks:
  size: 1200                      # 文本分块大小
  overlap: 100                    # 分块重叠

extract_graph:
  entity_types: [organization,person,geo,event]  # 实体类型
  max_gleanings: 1                # 最大提取轮数

community_reports:
  max_length: 2000                # 社区报告最大长度
```

## 🔧 常用命令

```bash
# 激活环境
conda activate graphrag-oneapi

# 构建索引
graphrag index --root .

# 本地搜索查询
graphrag query --root . --method local --query "你的问题"

# 全局搜索查询
graphrag query --root . --method global --query "你的问题"

# 启动 API 服务
python utils/main.py
```

## 📊 图数据库查询示例（Neo4j）

```cypher
-- 查看实体关系图
MATCH path = (:__Entity__)-[:RELATED]->(:__Entity__)
RETURN path LIMIT 200

-- 查看文档与文本块
MATCH (d:__Document__) WITH d LIMIT 1
MATCH path = (d)<-[:PART_OF]-(c:__Chunk__)
RETURN path LIMIT 100

-- 查看社区与实体
MATCH (c:__Community__) WITH c LIMIT 1
MATCH path = (c)<-[:IN_COMMUNITY]-()-[:RELATED]-(:__Entity__)
RETURN path LIMIT 100

-- 清空数据库
MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 25000 ROWS;
```

## 🤝 技术栈

- **GraphRAG**: 微软开源的图增强检索生成框架
- **FastAPI**: 高性能 Python Web 框架
- **LanceDB**: 向量数据库
- **Neo4j**: 图数据库
- **Plotly**: 交互式可视化库
- **NetworkX**: 图处理库
- **Scrapy**: 网页爬虫框架

## 📝 许可证

MIT License

## 👤 作者

**LiuJunDa**

- 日期: 2026-01-27
- 更新: 2026-02-04

---

> 💡 如有问题或建议，欢迎提交 Issue 或 Pull Request。
