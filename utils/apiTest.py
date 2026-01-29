"""
GraphRAG API 测试脚本 (适配 GraphRAG 2.7.0)

该脚本用于测试 GraphRAG 知识图谱问答系统的不同搜索模式。
支持全局搜索、本地搜索和综合搜索三种模式。

日期: 2026-01-28
作者: LiuJunDa
"""
import os
os.environ['no_proxy'] = 'localhost,127.0.0.1'

import requests
import json

BASE_URL = "http://localhost:8012"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"
HEALTH_URL = f"{BASE_URL}/health"
MODELS_URL = f"{BASE_URL}/v1/models"

headers = {"Content-Type": "application/json"}


def check_health():
    """检查服务健康状态"""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ 服务状态:")
            print(f"   - 版本: {data.get('version')}")
            print(f"   - GraphRAG: {data.get('graphrag_version')}")
            print(f"   - 本地搜索: {'就绪' if data.get('local_search_ready') else '未就绪'}")
            print(f"   - 全局搜索: {'就绪' if data.get('global_search_ready') else '未就绪'}")
            return True
        else:
            print(f"❌ 服务异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务，请确保服务已启动")
        return False


def list_models():
    """获取可用模型列表"""
    response = requests.get(MODELS_URL)
    if response.status_code == 200:
        models = response.json()['data']
        print("\n📋 可用模型:")
        for m in models:
            print(f"   - {m['id']}")


def test_search(model: str, query: str, stream: bool = False):
    """测试搜索功能"""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "stream": stream,
    }
    
    print(f"\n🔍 测试模型: {model}")
    print(f"   问题: {query[:50]}...")
    
    if stream:
        # 流式输出
        with requests.post(CHAT_URL, stream=True, headers=headers, data=json.dumps(data)) as response:
            print("   回答: ", end="")
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        try:
                            json_data = json.loads(line_str[6:])
                            content = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            pass
            print()  # 换行
    else:
        # 非流式输出
        response = requests.post(CHAT_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            print(f"   回答:\n{content[:500]}...")
        else:
            print(f"   ❌ 错误: {response.status_code} - {response.text}")


if __name__ == "__main__":
    print("=" * 60)
    print("GraphRAG API 测试 (v2.7.0)")
    print("=" * 60)
    
    # 1. 健康检查
    if not check_health():
        exit(1)
    
    # 2. 列出模型
    list_models()
    
    # 3. 测试查询
    # test_query = "韩立的名字是谁给起的？"
    test_query = "请介绍一下凡人修仙传前四章的主要内容。"
    
    # 测试全局搜索
    test_search("graphrag-global-search:latest", test_query)
    
    # 测试本地搜索
    # test_search("graphrag-local-search:latest", test_query)
    
    # 测试综合搜索
    # test_search("full-model:latest", test_query)
    
    # 测试流式输出
    # test_search("graphrag-local-search:latest", test_query, stream=True)