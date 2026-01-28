"""
GraphRAG 三维知识图谱可视化模块

该模块提供从 Parquet 文件中读取知识图谱数据并进行三维可视化的功能。
支持节点、边、度分布、中心性分布等多种可视化方式。

日期: 2026-01-27
作者: LiuJunDa
"""

import os  # 用于文件系统操作
import pandas as pd  # 用于数据处理和操作
import networkx as nx  # 用于创建和分析图结构
import plotly.graph_objects as go  # plotly：用于创建交互式可视化，plotly.graph_objects：用于创建低级的plotly图形对象
from plotly.subplots import make_subplots  # 用于创建子图
import plotly.express as px  # 用于快速创建统计图表


def read_parquet_files(directory):
    """
    读取指定目录下的所有 Parquet 文件并合并成一个 DataFrame
    
    使用 os.listdir 遍历目录，pd.read_parquet 读取每个文件，然后用 pd.concat 合并
    
    参数:
        directory: 包含 Parquet 文件的目录路径
    
    返回:
        合并后的 DataFrame，若无文件则返回空 DataFrame
    """
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            file_path = os.path.join(directory, filename)
            df = pd.read_parquet(file_path)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


def clean_dataframe(df):
    """
    清理 DataFrame，移除无效的行
    
    删除 source 和 target 列中的空值，将这两列转换为字符串类型
    
    参数:
        df: 需要清理的 DataFrame
    
    返回:
        清理后的 DataFrame
    """
    df = df.dropna(subset=['source', 'target'])
    df['source'] = df['source'].astype(str)
    df['target'] = df['target'].astype(str)
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

    node_trace.marker.color = node_adjacencies
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


def visualize_graph_plotly(G):
    """
    使用 Plotly 创建全面优化布局的高级交互式知识图谱可视化
    
    具体步骤:
        1. 创建 3D 布局
        2. 生成节点和边的轨迹
        3. 创建子图，包括 3D 图、度分布图和中心性分布图
        4. 添加交互式按钮和滑块
        5. 优化整体布局
    
    参数:
        G: networkx 图对象
    """

    if G.number_of_nodes() == 0:
        print("图为空。没有可视化内容。")
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

    fig.show()


def main():
    """
    主函数，协调整个程序的执行流程
    
    具体步骤:
        1. 读取 Parquet 文件
        2. 清理数据
        3. 创建知识图谱
        4. 打印图的统计信息
        5. 调用可视化函数
    """

    # 指定 Parquet 文件路径
    directory = 'D:/PythonWork/RAG/graph_test/ragtest/inputs/artifacts'
    # 读取指定目录下的所有 Parquet 文件并合并成一个 DataFrame
    df = read_parquet_files(directory)

    if df.empty:
        print("在指定目录中找不到数据。")
        return

    print("原始 DataFrame 形状:", df.shape)
    print("原始 DataFrame 列:", df.columns.tolist())
    print("原始 DataFrame 头部:")
    print(df.head())
    # 清理 DataFrame，移除无效的行
    df = clean_dataframe(df)

    print("\n清理后的 DataFrame 形状:", df.shape)
    print("清理后的 DataFrame 头部:")
    print(df.head())

    if df.empty:
        print("清理后没有有效数据。")
        return

    # 从 DataFrame 创建知识图谱
    G = create_knowledge_graph(df)

    print(f"\n图统计信息:")
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        # 将图 G 转换为无向图。如果 G 是有向图，转换为无向图后才能正确计算连通分量
        print(f"连通分量数: {nx.number_connected_components(G.to_undirected())}")
        # 对图 G 进行可视化
        visualize_graph_plotly(G)
    else:
        print("图为空。无法可视化。")


if __name__ == "__main__":
    main()
