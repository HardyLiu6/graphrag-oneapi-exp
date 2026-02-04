#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neo4j 图数据库导入脚本
====================

功能描述:
    将 GraphRAG 生成的 Parquet 文件数据导入到 Neo4j 图数据库中，
    构建包含文档、文本块、实体、关系、社区、协变量的知识图谱。

图谱结构:
    节点类型:
        - __Document__  : 原始文档
        - __Chunk__     : 文本块（文档切分后的片段）
        - __Entity__    : 实体（可附加类型标签，如 Person、Location 等）
        - __Community__ : 社区（实体聚类）
        - __Covariate__ : 协变量（声明/事件等附加信息）
        - Finding       : 社区发现（社区报告中的关键发现）

    关系类型:
        - PART_OF       : Chunk -> Document（文本块属于文档）
        - HAS_ENTITY    : Chunk -> Entity（文本块包含实体）
        - RELATED       : Entity -> Entity（实体间关系）
        - IN_COMMUNITY  : Entity -> Community（实体属于社区）
        - HAS_CHUNK     : Community -> Chunk（社区关联文本块）
        - HAS_FINDING   : Community -> Finding（社区包含发现）
        - HAS_COVARIATE : Chunk -> Covariate（文本块关联协变量）

Neo4j 常用查询示例:
    1. 查看实体关系图:
        MATCH path = (:__Entity__)-[:RELATED]->(:__Entity__)
        RETURN path LIMIT 200

    2. 查看文档与文本块:
        MATCH (d:__Document__) WITH d LIMIT 1
        MATCH path = (d)<-[:PART_OF]-(c:__Chunk__)
        RETURN path LIMIT 100

    3. 查看社区与实体:
        MATCH (c:__Community__) WITH c LIMIT 1
        MATCH path = (c)<-[:IN_COMMUNITY]-()-[:RELATED]-(:__Entity__)
        RETURN path LIMIT 100

    4. 清空数据库:
        MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 25000 ROWS;

依赖安装:
    pip install pandas neo4j-rust-ext

作者: LiuJunDa
日期: 2026-01-27
更新: 2026-02-04
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, cast, LiteralString

import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError


# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 默认配置（可通过环境变量覆盖）
# ============================================================================
DEFAULT_CONFIG = {
    "GRAPHRAG_FOLDER": os.getenv("GRAPHRAG_FOLDER", "/home/sunlight/Projects/graphrag-oneapi-exp/output"),
    "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:17687"),
    "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME", "neo4j"),
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "12345678"),
    "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE", "neo4j"),
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "1000")),
}


# ============================================================================
# Cypher 语句定义
# ============================================================================

# 数据库约束（确保数据唯一性和查询性能）
CONSTRAINT_STATEMENTS = [
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:__Document__) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:__Chunk__) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:__Community__) REQUIRE c.community IS UNIQUE",
    "CREATE CONSTRAINT covariate_title IF NOT EXISTS FOR (e:__Covariate__) REQUIRE e.title IS UNIQUE",
    "CREATE CONSTRAINT related_id IF NOT EXISTS FOR ()-[rel:RELATED]->() REQUIRE rel.id IS UNIQUE",
]

# 导入文档节点
DOCUMENT_STATEMENT = """
MERGE (d:__Document__ {id: value.id})
SET d += value {.title, .text}
"""

# 导入文本块节点并关联文档
CHUNK_STATEMENT = """
MERGE (c:__Chunk__ {id: value.id})
SET c += value {.text, .n_tokens}
WITH c, value
UNWIND value.document_ids AS document
MATCH (d:__Document__ {id: document})
MERGE (c)-[:PART_OF]->(d)
"""

# 导入实体节点并关联文本块
ENTITY_STATEMENT = """
MERGE (e:__Entity__ {id: value.id})
SET e += value {.title, .type, .description, .human_readable_id, .id, .text_unit_ids}
SET e.name = value.title
WITH e, value
CALL apoc.create.addLabels(e, 
    CASE WHEN coalesce(value.type, "") = "" 
    THEN [] 
    ELSE [apoc.text.upperCamelCase(replace(value.type, '"', ''))] 
    END
) YIELD node
UNWIND value.text_unit_ids AS text_unit
MATCH (c:__Chunk__ {id: text_unit})
MERGE (c)-[:HAS_ENTITY]->(e)
"""

# 导入实体间关系
RELATIONSHIP_STATEMENT = """
MATCH (source:__Entity__ {name: replace(value.source, '"', '')})
MATCH (target:__Entity__ {name: replace(value.target, '"', '')})
MERGE (source)-[rel:RELATED {id: value.id}]->(target)
SET rel += value {.weight, .human_readable_id, .description, .text_unit_ids}
RETURN count(*) AS createdRels
"""

# 导入社区报告节点
COMMUNITY_REPORT_STATEMENT = """
MERGE (c:__Community__ {id: value.id})
SET c += value {.community, .level, .title, .rank, .rating_explanation, .full_content, .summary}
WITH c, value
UNWIND range(0, size(value.findings) - 1) AS finding_idx
WITH c, value, finding_idx, value.findings[finding_idx] AS finding
MERGE (c)-[:HAS_FINDING]->(f:Finding {id: finding_idx})
SET f += finding
"""

# 导入社区节点并关联实体
COMMUNITY_STATEMENT = """
MERGE (c:__Community__ {community: value.id})
SET c += value {.level}
WITH *
UNWIND value.text_unit_ids AS text_unit_id
MATCH (t:__Chunk__ {id: text_unit_id})
MERGE (c)-[:HAS_CHUNK]->(t)
WITH *
UNWIND value.relationship_ids AS rel_id
MATCH (start:__Entity__)-[:RELATED {id: rel_id}]->(end:__Entity__)
MERGE (start)-[:IN_COMMUNITY]->(c)
MERGE (end)-[:IN_COMMUNITY]->(c)
RETURN count(DISTINCT c) AS createdCommunities
"""

# 导入协变量节点
COVARIATE_STATEMENT = """
MERGE (c:__Covariate__ {id: value.id})
SET c += apoc.map.clean(value, ["text_unit_id", "document_ids", "n_tokens"], [NULL, ""])
WITH c, value
MATCH (ch:__Chunk__ {id: value.text_unit_id})
MERGE (ch)-[:HAS_COVARIATE]->(c)
"""


# ============================================================================
# 核心功能类
# ============================================================================

class GraphRAGImporter:
    """
    GraphRAG 数据导入器
    
    负责将 GraphRAG 生成的 Parquet 文件批量导入到 Neo4j 图数据库，
    支持批处理、错误处理和进度显示。
    """
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        batch_size: int = 1000
    ):
        """
        初始化导入器
        
        参数:
            uri: Neo4j 连接地址（如 bolt://localhost:7687）
            username: 数据库用户名
            password: 数据库密码
            database: 目标数据库名称，默认为 "neo4j"
            batch_size: 每批导入的记录数，默认为 1000
        """
        self.uri = uri
        self.username = username
        self.database = database
        self.batch_size = batch_size
        self.driver = None
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            logger.info(f"✓ 成功连接到 Neo4j: {uri}")
        except ServiceUnavailable as e:
            logger.error(f"✗ 无法连接到 Neo4j 服务器: {e}")
            raise
        except AuthError as e:
            logger.error(f"✗ Neo4j 认证失败，请检查用户名和密码: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 连接已关闭")
    
    def __enter__(self):
        """支持上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动关闭连接"""
        self.close()
    
    def execute_statement(self, statement: str) -> None:
        """
        执行单条 Cypher 语句
        
        参数:
            statement: 要执行的 Cypher 查询语句
        """
        if self.driver is None:
            raise RuntimeError("数据库连接未初始化")
        with self.driver.session(database=self.database) as session:
            session.run(cast(LiteralString, statement))
    
    def create_constraints(self) -> None:
        """创建数据库约束，确保节点唯一性并优化查询性能"""
        logger.info("正在创建数据库约束...")
        for statement in CONSTRAINT_STATEMENTS:
            try:
                self.execute_statement(statement)
                logger.debug(f"  约束创建成功: {statement[:60]}...")
            except Exception as e:
                # 约束可能已存在，忽略错误
                logger.debug(f"  约束跳过（可能已存在）: {e}")
        logger.info("✓ 数据库约束创建完成")
    
    def batched_import(
        self,
        statement: str,
        df: pd.DataFrame,
        description: str = "数据"
    ) -> int:
        """
        批量导入数据到 Neo4j
        
        将 DataFrame 分批次导入数据库，避免单次导入数据量过大导致内存问题。
        
        参数:
            statement: Cypher 导入语句（使用 value 变量引用每行数据）
            df: 要导入的 Pandas DataFrame
            description: 数据描述，用于日志显示
        
        返回:
            成功导入的总行数
        """
        total = len(df)
        if total == 0:
            logger.warning(f"  {description} 数据为空，跳过导入")
            return 0
        
        start_time = time.time()
        imported = 0
        
        logger.info(f"开始导入 {description}（共 {total} 条）...")
        
        if self.driver is None:
            raise RuntimeError("数据库连接未初始化")
        
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = df.iloc[start:end]
            
            try:
                query_text = cast(LiteralString, "UNWIND $rows AS value " + statement)
                result = self.driver.execute_query(
                    query_text,
                    rows=batch.to_dict('records'),
                    database_=self.database
                )
                imported += len(batch)
                
                # 显示进度和统计
                progress = imported / total * 100
                counters = result.summary.counters
                logger.info(f"  进度: {imported}/{total} ({progress:.1f}%) - {counters}")
                
            except Exception as e:
                logger.error(f"  批次导入失败 [{start}:{end}]: {e}")
                raise
        
        elapsed = time.time() - start_time
        logger.info(f"✓ {description}: {total} 条记录导入完成，耗时 {elapsed:.2f} 秒")
        return total


def check_parquet_file(folder: str, filename: str) -> Optional[Path]:
    """
    检查 Parquet 文件是否存在
    
    参数:
        folder: 文件夹路径
        filename: 文件名（如 documents.parquet）
    
    返回:
        Path 对象（文件存在时）或 None（文件不存在时）
    """
    filepath = Path(folder) / filename
    if filepath.exists():
        return filepath
    logger.warning(f"  文件不存在，跳过: {filepath}")
    return None


def import_all_data(importer: GraphRAGImporter, data_folder: str) -> dict:
    """
    执行完整的数据导入流程
    
    按照依赖顺序导入所有数据：
    文档 -> 文本块 -> 实体 -> 关系 -> 社区报告 -> 社区 -> 协变量
    
    参数:
        importer: GraphRAGImporter 实例
        data_folder: Parquet 文件所在目录
    
    返回:
        包含各类数据导入数量的字典
    """
    stats = {}
    
    # 步骤 1: 创建约束
    importer.create_constraints()
    
    # 步骤 2: 导入文档
    logger.info("-" * 50)
    if filepath := check_parquet_file(data_folder, "documents.parquet"):
        df = pd.read_parquet(filepath, columns=["id", "title", "text"])
        stats["文档"] = importer.batched_import(DOCUMENT_STATEMENT, df, "文档")
    
    # 步骤 3: 导入文本块
    logger.info("-" * 50)
    if filepath := check_parquet_file(data_folder, "text_units.parquet"):
        df = pd.read_parquet(filepath, columns=[
            "id", "text", "n_tokens", "document_ids", 
            "entity_ids", "relationship_ids", "covariate_ids"
        ])
        stats["文本块"] = importer.batched_import(CHUNK_STATEMENT, df, "文本块")
    
    # 步骤 4: 导入实体
    logger.info("-" * 50)
    if filepath := check_parquet_file(data_folder, "entities.parquet"):
        df = pd.read_parquet(filepath, columns=[
            "title", "type", "description", "human_readable_id", "id", "text_unit_ids"
        ])
        stats["实体"] = importer.batched_import(ENTITY_STATEMENT, df, "实体")
    
    # 步骤 5: 导入关系
    logger.info("-" * 50)
    if filepath := check_parquet_file(data_folder, "relationships.parquet"):
        df = pd.read_parquet(filepath, columns=[
            "source", "target", "id", "weight", 
            "human_readable_id", "description", "text_unit_ids"
        ])
        stats["关系"] = importer.batched_import(RELATIONSHIP_STATEMENT, df, "关系")
    
    # 步骤 6: 导入社区报告
    logger.info("-" * 50)
    if filepath := check_parquet_file(data_folder, "community_reports.parquet"):
        df = pd.read_parquet(filepath, columns=[
            "id", "community", "findings", "title", "summary",
            "level", "rank", "rating_explanation", "full_content"
        ])
        stats["社区报告"] = importer.batched_import(COMMUNITY_REPORT_STATEMENT, df, "社区报告")
    
    # 步骤 7: 导入社区
    logger.info("-" * 50)
    if filepath := check_parquet_file(data_folder, "communities.parquet"):
        df = pd.read_parquet(filepath, columns=[
            "id", "level", "title", "text_unit_ids", "relationship_ids"
        ])
        stats["社区"] = importer.batched_import(COMMUNITY_STATEMENT, df, "社区")
    
    # 步骤 8: 导入协变量
    logger.info("-" * 50)
    if filepath := check_parquet_file(data_folder, "covariates.parquet"):
        df = pd.read_parquet(filepath)
        stats["协变量"] = importer.batched_import(COVARIATE_STATEMENT, df, "协变量")
    
    return stats


def print_summary(stats: dict) -> None:
    """
    打印导入摘要
    
    参数:
        stats: 包含各类数据导入数量的字典
    """
    logger.info("=" * 50)
    logger.info("📊 导入完成！统计信息:")
    logger.info("=" * 50)
    total = 0
    for name, count in stats.items():
        logger.info(f"  • {name}: {count} 条")
        total += count
    logger.info("-" * 50)
    logger.info(f"  总计: {total} 条记录")
    logger.info("=" * 50)


def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace 对象
    """
    parser = argparse.ArgumentParser(
        description="GraphRAG 数据导入 Neo4j 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 使用默认配置运行
    python neo4jTest.py

    # 指定数据目录
    python neo4jTest.py --folder /path/to/output

    # 指定 Neo4j 连接参数
    python neo4jTest.py --uri bolt://localhost:7687 --password mypassword

    # 显示详细日志
    python neo4jTest.py -v

环境变量:
    GRAPHRAG_FOLDER  - Parquet 文件目录
    NEO4J_URI        - Neo4j 连接地址
    NEO4J_USERNAME   - Neo4j 用户名
    NEO4J_PASSWORD   - Neo4j 密码
    NEO4J_DATABASE   - Neo4j 数据库名
    BATCH_SIZE       - 批处理大小
        """
    )
    parser.add_argument(
        "--folder", "-f",
        default=DEFAULT_CONFIG["GRAPHRAG_FOLDER"],
        help=f"Parquet 文件目录路径（默认: {DEFAULT_CONFIG['GRAPHRAG_FOLDER']}）"
    )
    parser.add_argument(
        "--uri", "-u",
        default=DEFAULT_CONFIG["NEO4J_URI"],
        help=f"Neo4j 连接 URI（默认: {DEFAULT_CONFIG['NEO4J_URI']}）"
    )
    parser.add_argument(
        "--username",
        default=DEFAULT_CONFIG["NEO4J_USERNAME"],
        help=f"Neo4j 用户名（默认: {DEFAULT_CONFIG['NEO4J_USERNAME']}）"
    )
    parser.add_argument(
        "--password", "-p",
        default=DEFAULT_CONFIG["NEO4J_PASSWORD"],
        help="Neo4j 密码"
    )
    parser.add_argument(
        "--database", "-d",
        default=DEFAULT_CONFIG["NEO4J_DATABASE"],
        help=f"Neo4j 数据库名称（默认: {DEFAULT_CONFIG['NEO4J_DATABASE']}）"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_CONFIG["BATCH_SIZE"],
        help=f"批处理大小（默认: {DEFAULT_CONFIG['BATCH_SIZE']}）"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志（DEBUG 级别）"
    )
    return parser.parse_args()


def main():
    """主函数入口"""
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查数据目录是否存在
    if not Path(args.folder).exists():
        logger.error(f"✗ 数据目录不存在: {args.folder}")
        sys.exit(1)
    
    logger.info("=" * 50)
    logger.info("🚀 GraphRAG -> Neo4j 数据导入工具")
    logger.info("=" * 50)
    logger.info(f"数据目录: {args.folder}")
    logger.info(f"Neo4j 地址: {args.uri}")
    logger.info(f"目标数据库: {args.database}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info("=" * 50)
    
    try:
        # 使用上下文管理器确保连接正确关闭
        with GraphRAGImporter(
            uri=args.uri,
            username=args.username,
            password=args.password,
            database=args.database,
            batch_size=args.batch_size
        ) as importer:
            stats = import_all_data(importer, args.folder)
            print_summary(stats)
            
    except KeyboardInterrupt:
        logger.warning("\n用户中断操作")
        sys.exit(130)
    except Exception as e:
        logger.error(f"✗ 导入失败: {e}")
        sys.exit(1)
    
    logger.info("🎉 全部完成！")


if __name__ == "__main__":
    main()










