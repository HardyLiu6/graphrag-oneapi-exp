"""
GraphRAG 三维知识图谱可视化模块
===================================================

功能描述:
    从 GraphRAG 2.x 处理系统的 Parquet 文件中读取知识图谱数据
    并进行三维交互式可视化。

支持的可视化:
    - 3D 网络图（节点、边、标签）
    - 节点度分布直方图
    - 节点中心性箱线图
    - 两種布局子图展示

依赖:
    pandas, networkx, plotly, plotly-express

作者: LiuJunDa
日期: 2026-01-27
更新: 2026-02-04 (GraphRAG 2.x 兼容)
"""

import os
import sys
import argparse
import logging
import tempfile
import http.server
import socketserver
from pathlib import Path

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def read_parquet_files(directory):
    """
    读取指定目录下的 Parquet 文件
    
    GraphRAG 2.x 输出数据格式:
        - entities.parquet: 帮助表（id, title, type, description, ...）
        - relationships.parquet: 关系表（source, target, id, weight, ...）
        - communities.parquet: 社区表（id, community, title, ...）
    
    此函数需要 relationships.parquet 为図数据源。
    
    参数:
        directory: 包含 Parquet 文件的目录路径
    
    返回:
        pandas.DataFrame - 测试 relationships.parquet 并浅试提取 source/target 列
    """
    rel_file = Path(directory) / 'relationships.parquet'
    
    if not rel_file.exists():
        logger.warning(f"资源需求文件不存在: {rel_file}")
        logger.info("正在查找可用的 Parquet 文件...")
        parquet_files = list(Path(directory).glob('*.parquet'))
        if not parquet_files:
            logger.error(f"在 {directory} 中找不到 Parquet 文件")
            return pd.DataFrame()
        
        # 尝试第一个 parquet 文件
        df = pd.read_parquet(parquet_files[0])
        logger.info(f"使用 {parquet_files[0].name}, 列名: {df.columns.tolist()}")
        return df
    
    df = pd.read_parquet(rel_file)
    logger.info(f"成功加载 relationships.parquet, 子数: {len(df)}")
    return df





def clean_dataframe(df):
    """
    清理 DataFrame
    
    为了增强匠强性，房底粗曝的一些担惧:
        - 刪除 source/target 列为空的记录
        - 输入两列为字符串类型
        - 移除空白值输入
    
    参数:
        df: 原始 DataFrame
    
    返回:
        清理后的 DataFrame
    """
    original_count = len(df)
    
    # 检查是否存在 source/target 列
    if 'source' not in df.columns or 'target' not in df.columns:
        logger.warning(f"数据表缺少 'source' 或 'target' 列，可用列: {df.columns.tolist()}")
        # 尝试找不到也有 title/name 等一些列
        if 'description' in df.columns:
            logger.info("将 'description' 的第一个记录用作测试")
            return df.head(1)
        return df
    
    # 刪除空值记录
    df = df.dropna(subset=['source', 'target'])
    
    # 一些空值可能是空字符串
    df = df[(df['source'].astype(str).str.strip() != '') & (df['target'].astype(str).str.strip() != '')]
    
    # 输入两列为字符串类型
    df['source'] = df['source'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()
    
    removed_count = original_count - len(df)
    logger.info(f"数据清理: 去除 {removed_count} 条记录, 保留 {len(df)} 条")
    
    return df





def create_knowledge_graph(df):
    """
    从 DataFrame 创建知识图谱
    
    使用 networkx 创建有向图，遍历 DataFrame 的每一行，添加边和属性
    
    参数:
        df: 包含 source、target 和其他属性的 DataFrame
    
    返回:
        创建好的 networkx 有向图
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        source = row['source']
        target = row['target']
        attributes = {k: v for k, v in row.items() if k not in ['source', 'target']}
        G.add_edge(source, target, **attributes)
    return G


def create_node_link_trace(G, pos):
    """
    创建节点和边的 3D 轨迹
    
    使用 networkx 的布局信息创建 Plotly 的 Scatter3d 对象
    
    参数:
        G: networkx 图对象
        pos: 节点位置字典
    
    返回:
        (edge_trace, node_trace) - 边轨迹和节点轨迹的元组
    """
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_z = [pos[node][2] for node in G.nodes()]

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(text='节点连接数', side='right'),
                xanchor='left'
                
            )
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f'节点: {node}<br>连接数: {len(adjacencies)}')

    # 设置标记颜色和文本
    marker = node_trace.marker
    if marker is not None:
        marker.color = node_adjacencies  # type: ignore
    node_trace.text = node_text

    return edge_trace, node_trace


def create_edge_label_trace(G, pos, edge_labels):
    """
    创建边标签的 3D 轨迹
    
    计算边的中点位置，创建 Scatter3d 对象显示标签
    
    参数:
        G: networkx 图对象
        pos: 节点位置字典
        edge_labels: 边标签字典
    
    返回:
        边标签轨迹对象
    """
    return go.Scatter3d(
        x=[pos[edge[0]][0] + (pos[edge[1]][0] - pos[edge[0]][0]) / 2 for edge in edge_labels],
        y=[pos[edge[0]][1] + (pos[edge[1]][1] - pos[edge[0]][1]) / 2 for edge in edge_labels],
        z=[pos[edge[0]][2] + (pos[edge[1]][2] - pos[edge[0]][2]) / 2 for edge in edge_labels],
        mode='text',
        text=list(edge_labels.values()),
        textposition='middle center',
        hoverinfo='none'
    )


def create_degree_distribution(G):
    """
    创建节点度分布直方图
    
    使用 plotly.express 创建直方图
    
    参数:
        G: networkx 图对象
    
    返回:
        Plotly 图形对象
    """
    degrees = [d for n, d in G.degree()]
    fig = px.histogram(x=degrees, nbins=20, labels={'x': '度数', 'y': '数量'})
    fig.update_layout(
        title_text='节点度分布',
        margin=dict(l=0, r=0, t=30, b=0),
        height=300
    )
    return fig


def create_centrality_plot(G):
    """
    创建节点中心性分布箱线图
    
    计算度中心性，使用 plotly.express 创建箱线图
    
    参数:
        G: networkx 图对象
    
    返回:
        Plotly 图形对象
    """
    centrality = nx.degree_centrality(G)
    centrality_values = list(centrality.values())
    fig = px.box(y=centrality_values, labels={'y': '中心性'})
    fig.update_layout(
        title_text='度中心性分布',
        margin=dict(l=0, r=0, t=30, b=0),
        height=300
    )
    return fig


def start_http_server(html_content, port=0):
    """
    启动一个简单的 HTTP 服务器来托管 HTML 内容
    
    参数:
        html_content: HTML 文件内容
        port: 端口号（0 表示自动选择可用端口）
    
    返回:
        (httpd, port, temp_dir): 服务器对象、实际端口、临时目录
    """
    # 创建临时目录存放 HTML 文件
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / 'index.html'
    temp_file.write_text(html_content, encoding='utf-8')
    
    # 切换到临时目录
    os.chdir(temp_dir)
    
    # 自定义 Handler，抑制日志输出
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # 不输出访问日志
    
    # 启动服务器
    httpd = socketserver.TCPServer(("", port), QuietHandler)
    actual_port = httpd.server_address[1]
    
    return httpd, actual_port, temp_dir


def visualize_graph_plotly(G, output_file=None, serve=False, port=8050):
    """
    使用 Plotly 创建全面优化布局的高级交互式知识图谱可视化
    
    具体步骤:
        1. 创建 3D 布局
        2. 生成节点和边的轨迹
        3. 创建子图，包括 3D 图、度分布图和中心性分布图
        4. 添加交互式按钮和滑块
        5. 保存或显示结果
    
    参数:
        G: networkx 图对象
        output_file: 输出 HTML 文件路径（不提供则需要浏览器打开）
        serve: 是否启动 HTTP 服务器（适用于 WSL2 环境）
        port: HTTP 服务器端口（默认 8050，0 表示自动选择）
    """
    if G.number_of_nodes() == 0:
        logger.error("图为空。没有可视化内容。")
        return

    pos = nx.spring_layout(G, dim=3)  # 3D 布局
    edge_trace, node_trace = create_node_link_trace(G, pos)

    edge_labels = nx.get_edge_attributes(G, 'relation')
    edge_label_trace = create_edge_label_trace(G, pos, edge_labels)

    degree_dist_fig = create_degree_distribution(G)
    centrality_fig = create_centrality_plot(G)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.7, 0.3],
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}]
        ],
        subplot_titles=("GraphRAG 的三维知识图谱", "节点度分布", "度中心性分布")
    )

    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    fig.add_trace(edge_label_trace, row=1, col=1)

    fig.add_trace(degree_dist_fig.data[0], row=1, col=2)
    fig.add_trace(centrality_fig.data[0], row=2, col=2)

    # 更新 3D 布局
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title='', backgroundcolor='rgb(255,255,255)'),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title='', backgroundcolor='rgb(255,255,255)'),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title='', backgroundcolor='rgb(255,255,255)'),
            aspectmode='cube',
            # bgcolor='rgb(0,0,0)'  # 设置背景颜色
        ),
        # paper_bgcolor='rgb(0,0,0)',  # 设置图表纸张背景颜色
        # plot_bgcolor='rgb(0,0,0)',  # 设置绘图区域背景颜色
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )

    # 为不同的布局添加按钮
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(args=[{"visible": [True, True, True, True, True]}], label="显示全部", method="update"),
                    dict(args=[{"visible": [True, True, False, True, True]}], label="隐藏边标签",
                         method="update"),
                    dict(args=[{"visible": [False, True, False, True, True]}], label="仅显示节点", method="update")
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # 为节点大小添加滑块
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "节点大小: "},
            pad={"t": 50},
            steps=[dict(method='update',
                        args=[{'marker.size': [i] * len(G.nodes())}],
                        label=str(i)) for i in range(5, 21, 5)]
        )]
    )

    # 优化整体布局
    # fig.update_layout(
    #     height=1198,  # 增加整体高度
    #     width=2055,  # 增加整体宽度
    #     title_text="高级交互式知识图谱",
    #     margin=dict(l=10, r=10, t=25, b=10),
    #     legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    # )

    # 保存或显示结果
    if output_file:
        fig.write_html(output_file)
        logger.info(f"✅ 可视化已保存到: {output_file}")
        logger.info("成功! 可以打开此文件查看交互式图表")
    elif serve:
        # WSL2 模式：启动 HTTP 服务器
        logger.info("🌐 启动 HTTP 服务器模式 (WSL2 兼容)...")
        html_content = fig.to_html(include_plotlyjs=True, full_html=True)
        
        original_dir = os.getcwd()
        try:
            httpd, actual_port, temp_dir = start_http_server(html_content, port)
            
            url = f"http://localhost:{actual_port}"
            logger.info("=" * 60)
            logger.info("🚀 服务器已启动!")
            logger.info(f"📍 请在 Windows 浏览器中打开: {url}")
            logger.info("=" * 60)
            logger.info("按 Ctrl+C 停止服务器")
            
            # 尝试自动打开浏览器 (WSL2 可以通过 wslview 或 explorer.exe 打开)
            try:
                # 尝试使用 Windows 的 explorer.exe 打开 URL
                os.system(f'explorer.exe "{url}" 2>/dev/null || xdg-open "{url}" 2>/dev/null &')
            except Exception:
                pass  # 忽略错误，用户可以手动打开
            
            # 运行服务器
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\n👋 服务器已停止")
        finally:
            os.chdir(original_dir)
            # 清理临时文件
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    else:
        logger.info("正在打开浏览器...")
        try:
            fig.show()
        except Exception as e:
            logger.warning(f"自动打开浏览器失败: {e}")
            logger.info("提示: 在 WSL2 中请使用 --serve 参数启动 HTTP 服务器")
            logger.info("      或使用 --output 参数保存为 HTML 文件")


def main():
    """
    主函数 - 协调整个程序的执行流程
    
    流程:
        1. 解析命令行参数
        2. 读取 Parquet 文件
        3. 清理数据
        4. 创建知识图谱
        5. 打印统计信息
        6. 可视化图谱
    """
    # 命令行参数配置
    parser = argparse.ArgumentParser(
        description='GraphRAG 三维知识图谱可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python graphrag3dknowledge.py
    python graphrag3dknowledge.py -d /path/to/graphrag/output
    python graphrag3dknowledge.py -d output -v

支持的文件格式:
    - relationships.parquet (推荐)
    - 任何包含 source/target 列的 parquet 文件
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        default='/home/sunlight/Projects/graphrag-oneapi-exp/output',
        help='GraphRAG 输出数据目录（默认: ./output）'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细日志'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='保存为 HTML 文件，例如: output.html'
    )
    parser.add_argument(
        '--serve', '-s',
        action='store_true',
        help='启动 HTTP 服务器模式（推荐 WSL2 环境使用）'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8050,
        help='HTTP 服务器端口（默认: 8050，0 表示自动选择）'
    )
    parser.add_argument(
        '--min-nodes',
        type=int,
        default=5,
        help='最小节点数以进行可视化（默认: 5）'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("🚀 GraphRAG 三维知识图谱可视化工具")
    logger.info("=" * 60)
    logger.info(f"数据目录: {args.directory}")
    
    # 检查目录是否存在
    if not Path(args.directory).exists():
        logger.error(f"✗ 目录不存在: {args.directory}")
        sys.exit(1)
    
    # 读取指定目录下的 Parquet 文件
    logger.info("正在读取数据...")
    df = read_parquet_files(args.directory)
    
    if df.empty:
        logger.error("✗ 无法读取数据")
        sys.exit(1)
    
    logger.info(f"✓ 原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    logger.debug(f"  列名: {df.columns.tolist()}")
    logger.debug(f"  数据预览:\n{df.head(2)}")
    
    # 清理 DataFrame
    logger.info("正在清理数据...")
    df = clean_dataframe(df)
    
    if df.empty:
        logger.error("✗ 清理后没有有效数据")
        sys.exit(1)
    
    # 创建知识图谱
    logger.info("正在构建知识图谱...")
    G = create_knowledge_graph(df)
    
    logger.info("=" * 60)
    logger.info("📊 图谱统计:")
    logger.info("=" * 60)
    logger.info(f"  节点数: {G.number_of_nodes()}")
    logger.info(f"  边数: {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        undirected = G.to_undirected()
        logger.info(f"  连通分量数: {nx.number_connected_components(undirected)}")
        logger.info(f"  平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    logger.info("=" * 60)
    
    # 可视化
    if G.number_of_nodes() >= args.min_nodes:
        logger.info(f"✓ 节点数 ({G.number_of_nodes()}) >= 最小要求 ({args.min_nodes})，开始可视化...")
        
        # 确定输出模式
        if args.serve:
            # HTTP 服务器模式
            visualize_graph_plotly(G, serve=True, port=args.port)
        elif args.output:
            # 保存到文件
            visualize_graph_plotly(G, output_file=args.output)
        else:
            # 默认：保存为 HTML 文件（WSL2 环境下更可靠）
            dir_name = Path(args.directory).name or 'graph'
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"graph_3d_{dir_name}_{timestamp}.html"
            visualize_graph_plotly(G, output_file=output_file)
    else:
        logger.warning(f"⚠ 节点数 ({G.number_of_nodes()}) < 最小要求 ({args.min_nodes})，跳过可视化")
        logger.warning(f"  可使用 --min-nodes {G.number_of_nodes()} 来强制可视化")


if __name__ == "__main__":
    main()
