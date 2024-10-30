# -*- coding: utf-8 -*-


import json
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from flask_caching import Cache
from dash import dash_table
import os
import logging
import time
import re
import base64


# 定义允许的模型列表
allowed_models = ['vllm', 'trt', 'sglang', 'lightllm', 'ppl', 'lmdeploy']

# 网页上传数据使用，验证文件名格式
def validate_filename(filename):
    # 先检查文件名是否符合正则表达式格式
    pattern = r'^([a-zA-Z]+)##[a-zA-Z0-9\.\-]+##[a-zA-Z0-9\-]+##tp_\d+##inputlen_[a-zA-Z0-9]+##outputlen_\d+##clients_\d+##device_[a-zA-Z0-9]+\.json$'
    match = re.match(pattern, filename)
    
    if not match:
        return False, "Filename must match backend##model##dtype##tp_?##inputlen_?##outputlen_?##clients_?##device_?.json"

    # 提取文件名的第一个部分（模型名称），并检查是否在允许的模型列表中
    model_name = match.group(1)
    if model_name not in allowed_models:
        return False, f"Model '{model_name}' is not allowed. Must be one of {allowed_models}."
    
    return True, None

def parse_file(file_path):
    # 从文件名中提取信息
    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    parts = file_name_no_ext.split("##")
    if len(parts) != 8:
        print(file_path)
        raise ValueError("Filename does not match the expected format.")
    backend = parts[0]
    model = parts[1]
    dtype = parts[2]
    tp = parts[3].split("_")[1]
    input_len = parts[4].split("_")[1]
    output_len = int(parts[5].split("_")[1])
    clients = int(parts[6].split("_")[1])
    device = parts[7].split("_")[1].split(".")[0]  # H100
    
    # 读取文件内容
    with open(file_path, 'r') as f:
        content = json.load(f)
    # print(content)
    # 创建一个字典来存储所有信息
    data = {
        'Device' : device,
        'Backend': backend,
        'Model': model,
        'Dtype':dtype,
        'TP': tp,
        # 'Input Length': input_len,
        # 'Output Length': output_len,
        'Clients': clients,
        'Total QPS': content['Total QPS'],
        'P25 TTFT (s)': content['first_token_time_dict']['P25'],
        'P50 TTFT (s)': content['first_token_time_dict']['P50'],
        'P99 TTFT (s)': content['first_token_time_dict']['P99'],
        'P25 ITL (ms)': content['decode_token_time_dict']['P25'],
        'P50 ITL (ms)': content['decode_token_time_dict']['P50'],
        'P99 ITL (ms)': content['decode_token_time_dict']['P99'],
        'Input-Output': f"{input_len}-{output_len}"
    }

    return data

# 假设所有文件都在一个目录下
json_directory_path = '/nvme/sangchengmeng/result_save3/'  # 替换为实际文件路径
json_file_paths = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(json_directory_path)
    for file in files
    if file.endswith('.json')
]

# 使用上面定义的函数解析每个文件并构建 DataFrame
json_data_list = [parse_file(file_path) for file_path in json_file_paths]

df = pd.DataFrame(json_data_list)


def read_md_file(md_file_path):
# 读取Markdown文件
    logging.info(f"Reading file: {md_file_path}")
    # md_file_path = '/container/nvme/sangchengmeng/mistral-12B.md'  # 替换为你的Markdown文件路径
    with open(md_file_path, 'r') as file:
        lines = file.readlines()

    # 提取 `test model` 名称
    test_model_line = lines[0]
    test_model = test_model_line.split(":")[1].strip()
    
    # 定位表格数据的开始和结束
    start_index = None
    for i, line in enumerate(lines):
        if line.startswith('|mode|'):  # 定位表头行
            start_index = i
            break
    if start_index is not None:
        table_data = lines[start_index:]  # 提取从表格开始行到文件结尾的所有行

        # 使用pandas将表格数据转换为DataFrame
        # 注意：我们需要去掉多余的竖线和空格
        clean_data = [line.strip().strip('|').split('|') for line in table_data]
        headers = [h.strip() for h in clean_data[0]]  # 提取列名
        data_rows = clean_data[2:]  # 提取数据行（跳过分隔符行）

        md_df = pd.DataFrame(data_rows, columns=headers)
        md_df['test_model'] = test_model

        # 将数据类型从字符串转换为适当的类型，例如数值
        # 转换数值列
        numeric_cols = ['world_size', 'batch_size', 'input_len', 'output_len', 'prefill_cost', 'first_step_latency',
                    'last_step_latency', 'mean_latency', 'prefill_throughput', 'decode_throughput', 'total_throughput',
                    'card_num_per_qps']
        for col in numeric_cols:
            md_df[col] = pd.to_numeric(md_df[col], errors='coerce')
    return md_df

md_directory = '/nvme/sangchengmeng/result_save3/' 
md_files = [os.path.join(root, file) for root, dirs, files in os.walk(md_directory) for file in files if file.endswith('.md')]

print(md_files)
df_list = [read_md_file(md_file) for md_file in md_files]
combined_df = pd.concat(df_list, ignore_index=True)

models = combined_df['test_model'].unique()
combined_df['test_model_lower'] = combined_df['test_model'].str.lower()
# print(combined_df)

world_sizes = combined_df['world_size'].unique()
batch_sizes = combined_df['batch_size'].unique()
input_lens = combined_df['input_len'].unique()
output_lens = combined_df['output_len'].unique()
devices = combined_df['devices'].unique()




display_columns = ['backend','mode', 'prefill_cost', 'first_step_latency', 
                        'last_step_latency', 'mean_latency', 'prefill_throughput',
                        'decode_throughput', 'total_throughput', 'card_num_per_qps']

print(df)
df['Model_lower'] = df['Model'].str.lower()
df['Backend_lower'] = df['Backend'].str.lower()

# 创建 Dash 应用
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.enable_dev_tools(debug=False, dev_tools_hot_reload=False)
server = app.server
# 配置缓存
cache = Cache(server, config={
    'CACHE_TYPE': 'simple',  # 使用简单的内存缓存
    'CACHE_DEFAULT_TIMEOUT': 300  # 缓存超时时间为300秒
})


# 应用布局
app.layout = dbc.Container([
    dbc.Row([  # 创建一行
    dbc.Col([
        html.H1("Model Inference Performance", className="text-left mt-4 mb-4"),  # 标题放在左边
    ] ,lg={"size": "auto", "offset": 4}),  # 根据内容自适应宽度
    
    dbc.Col([  # 上传文件组件放在右边
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                '上传数据',
            ]),
            style={
            'width': '180px',
            'height': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '10px',
            'textAlign': 'center',
            'display': 'flex',  # 使用 flex 布局
            'justifyContent': 'center',  # 水平居中
            'alignItems': 'center',  # 垂直居中
            'margin': '10px',
            'margin-left': '-30px',  # 向左移动
            },
            multiple=True  # 如果你想支持多文件上传
        ),
            html.Div(id='output-data-upload'),  # 用于显示上传的状态
            dcc.ConfirmDialog(  # 弹出框
            id='confirm-dialog_successd',
            message='文件上传成功，请刷新页面！',
        ),dcc.Location(id='page-refresh'),
            dcc.Store(id='upload-status', data={'uploaded': False}),
        ], width="auto", className="d-flex align-items-center justify-content-end")  # 设置右边的布局和对齐方式
    ], justify="between"),  # 确保标题和上传组件在一行并两端对齐
    
        # 提示命名不符合规范的弹出框
    dcc.ConfirmDialog(
        id='confirm-dialog_failed',
        message='文件名不符合规范，格式必须为\nbackend##model##tp_?##inputlen_?##outputlen_?##clients_?##device_?.json，\n而且backend必须是vllm, trt, sglang, lightllm, ppl, lmdeploy中的一个.',
    ),
    # 添加多个筛选框
    
# 将两个卡片并排放置在同一行
dbc.Row([
    # 为Quantization添加卡片，放在最左边
    dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.Label("quantization:(set None to ignore)", className="font-weight-bold"),
                dcc.Dropdown(
                    id='backend-dropdown',
                    options=[
                        {'label': 'trt', 'value': 'trt'},
                        {'label': 'vllm', 'value': 'vllm'},
                        {'label': 'lmdeploy', 'value': 'lmdeploy'},
                        {'label': 'None', 'value': None}
                    ],
                    value=None,  # 默认值为None
                    className="mb-2",
                    style={'width': '100%'}
                ),
            ], style={"backgroundColor": "rgba(255, 255, 255, 0.0)", "border": "none", "box-shadow": "none"})
        ], className="mb-4", style={"box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"}),
        width=2
    ),

    # 为其余6个筛选框添加卡片，放在右侧
    dbc.Col(
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    
                    dbc.Col([
                        html.Label("Select Model:", className="font-weight-bold"),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[{'label': model, 'value': model} for model in sorted(df['Model'].unique())],
                            value=df['Model'].unique()[0],
                            className="mb-2",
                            style={'width': '80%'}
                        ),
                    ], width=2),
                    
                    dbc.Col([
                        html.Label("Select TP:", className="font-weight-bold"),
                        dcc.Dropdown(
                            id='tp-dropdown',
                            options=[{'label': tp, 'value': tp} for tp in df['TP'].unique()],
                            value=df['TP'].unique()[0],
                            className="mb-2",
                            style={'width': '80%'}
                        ),
                    ], width=2),
                    
                    dbc.Col([
                        html.Label("Select Device:", className="font-weight-bold"),
                        dcc.Dropdown(
                            id='device-dropdown',
                            options=[{'label': device, 'value': device} for device in df['Device'].unique()],
                            value=df['Device'].unique()[1],
                            className="mb-2",
                            style={'width': '80%'}
                        ),
                    ], width=2),
                    
                    
                    dbc.Col([
                        html.Label("Select Dtype:", className="font-weight-bold"),
                        dcc.Dropdown(
                            id='dtype-dropdown',
                            options=[{'label': dtype, 'value': dtype} for dtype in df['Dtype'].unique()],
                            value=df['Dtype'].unique()[0],
                            className="mb-2",
                            style={'width': '80%'}
                        ),
                    ], width=2),

                    dbc.Col([
                        html.Label("Select Clients:", className="font-weight-bold"),
                        dcc.Dropdown(
                            id='client-dropdown',
                            options=[{'label': client, 'value': client} for client in df['Clients'].unique()],
                            value=df['Clients'].unique()[0],
                            className="mb-2",
                            style={'width': '80%'}
                        ),
                    ], width=2),
                    
                    dbc.Col([
                        html.Label("Select Input-Output:", className="font-weight-bold"),
                        dcc.Dropdown(
                            id='input-output-dropdown',
                            options=[{'label': io, 'value': io} for io in df['Input-Output'].unique()],
                            value=df['Input-Output'].unique()[0],
                            className="mb-2",
                            style={'width': '100%'}
                        ),
                    ], width=2),
                    
                ], justify="center")
            ], style={"backgroundColor": "rgba(255, 255, 255, 0.0)", "border": "none", "box-shadow": "none"})
        ], className="mb-4", style={"box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"}),
        width=10
    )
], justify="start", style={"width": "100%", "margin": "0 auto"}),

    
    dbc.Row([
# 显示每个框架的启动参数
#         dbc.Col([
#             # 添加按钮和 Modal
# dbc.Button(
#         "显示框架启动命令",
#         id='show-commands-button',
#         n_clicks=0,
#         color="info",
#         className="mt-4",
#         style={
#             'background-color': 'white',  # 按钮背景颜色为白色
#             'color': 'black',  # 按钮文本颜色为黑色
#             'border': '2px solid black',  # 设置黑色边框
#             'border-radius': '5px',  # 设置圆角边框
#             'padding': '10px 20px',  # 调整内边距
#             'position': 'fixed',  # 将按钮固定在页面
#             'right': '105px',  # 距离右边 100px
#             'top': '0px',  # 距离底部 20px
#             'z-index': '1000'  # 确保按钮在其他元素之上
#         }
#     ),
#     dbc.Modal(
#         [
#             dbc.ModalHeader(dbc.ModalTitle("框架启动命令")),
#             dbc.ModalBody([
#                 dbc.Row([
#                     dbc.Col([
#                         html.H5("LightLLM"),
#                         html.P("使用以下命令启动 LightLLM:"),
#                         html.Code("lightllm_command --start", style={'display': 'block', 'background-color': '#f2f2f2', 'padding': '10px'})
#                     ], width=4, style={'border-right': '1px solid #ccc', 'padding-right': '10px'}),  # 左侧部分

#                     dbc.Col([
#                         html.H5("VLLM"),
#                         html.P("使用以下命令启动 VLLM:"),
#                         html.Code("vllm_command --start", style={'display': 'block', 'background-color': '#f2f2f2', 'padding': '10px'})
#                     ], width=4, style={'border-right': '1px solid #ccc', 'padding-right': '10px', 'padding-left': '10px'}),  # 中间部分

#                     dbc.Col([
#                         html.H5("LMDeploy"),
#                         html.P("使用以下命令启动 LMDeploy:"),
#                         html.Code("lmdeploy_command --start", style={'display': 'block', 'background-color': '#f2f2f2', 'padding': '10px'})
#                     ], width=4, style={'padding-left': '10px'}),  # 右侧部分
#                 ])
#             ]),
#             dbc.ModalFooter(
#                 dbc.Button("关闭", id="close-commands-modal", className="ml-auto")
#             ),
#         ],
#         id="commands-modal",
#         is_open=False,  # 初始状态为关闭
#     )
#         ], width=12),

    dbc.Row([
        dbc.Col(dcc.Loading(id="loading-1", type="default",
                            children=dcc.Graph(id='ITL-graph', config={'displayModeBar': False})),
                width=4, style={'padding-left': '100px'}),  # 设置图表列宽度
        dbc.Col(dcc.Loading(id="loading-2", type="default",
                            children=dcc.Graph(id='ttft-graph', config={'displayModeBar': False})),
                width=4, style={'padding-left': '60px', 'padding-right': '60px'}),  # 设置图表列宽度
        dbc.Col(dcc.Loading(id="loading-3", type="default",
                            children=dcc.Graph(id='qps-bar-chart', config={'displayModeBar': False})),
                width=4, style={'padding-right': '0px'}),  # 设置图表列宽度
    ]),

    html.Div(style={'height': '50px'}),  # 调整这个高度来控制表格与顶部内容的距离

    # 静态的筛选框
    dbc.Card([
        dbc.CardBody([

            html.H4(f"Select Filters to show model's static Inference Performance"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select world_size:", className="font-weight-bold"),
                    dcc.Dropdown(
                        id='world-size-dropdown',
                        options=[],
                        value=world_sizes[0],
                        className="mb-2",
                        style={'width': '80%'} 
                    ),
                ], width=2),

                dbc.Col([
                    html.Label("Select batch_size", className="font-weight-bold"),
                    dcc.Dropdown(
                        id='batch-size-dropdown',
                        options=[],
                        value=batch_sizes[0],
                        className="mb-2",
                        style={'width': '80%'} 
                    ),
                ], width=2),

                dbc.Col([
                    html.Label("Select Decive:", className="font-weight-bold"),
                    dcc.Dropdown(
                        id='devices-dropdown',
                        options=[],
                        value=devices[0],
                        className="mb-2",
                        style={'width': '80%'} 
                    ),
                ], width=2), 
                
                dbc.Col([
                    html.Label("Select input_len:", className="font-weight-bold"),
                    dcc.Dropdown(
                        id='input-len-dropdown',
                        options=[],
                        value=input_lens[0],
                        className="mb-2",
                        style={'width': '80%'} 
                    ),
                ], width=2),

                dbc.Col([
                    html.Label("Select output_len:", className="font-weight-bold"),
                    dcc.Dropdown(
                        id='output-len-dropdown',
                        options=[],
                        value=output_lens[0],
                        className="mb-2",
                        style={'width': '80%'} 
                    ),
                ], width=2),
                

            ],justify="center"  # 设置居中对齐
            ),
            html.Div(style={'height': '20px'}),
            dcc.Loading(id="loading-table", type="default",  # 包裹表格的加载组件
                        children=dash_table.DataTable(
                            id='md-table',
                            columns=[{"name": i, "id": i} for i in display_columns],
                            data=[],  # 初始无数据
                            page_size=10,  # 设置每页显示的行数
                            style_table={'overflowX': 'auto', 'width': '100%','maxHeight': '500px','margin': '0 auto'},
                            style_cell={'textAlign': 'center', 'padding': '8px 5px', 'vertical-align': 'middle','lineHeight': '1.5'},
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold',
                                'textAlign': 'center'
                            }
                        )
            )
        ],
        
         style={"backgroundColor": "rgba(255, 255, 255, 0.0)", "border": "none", "box-shadow": "none"})
    ], className="mb-4",style={"width": "90%", "margin": "0 auto", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"}),  # 添加卡片样式和外边距
    
    ])
], fluid=True)


# @cache.memoize()
def get_filtered_data(model, dtype, tp, input_output, clients, device):
    # 根据给定的过滤条件查询 DataFrame
    model_select = model.lower()
    result = df[
        (df['Model_lower'] == model_select) & 
        (df['Dtype'] == dtype) & 
        (df['TP'] == tp) &
        (df['Input-Output'] == input_output) &
        (df['Clients'] == clients) &
        (df['Device'] == device)
    ]
    return result


# @cache.memoize()
def get_filtered_data_dtype(backend, model, tp, device, clients, input_output):
    # 根据给定的过滤条件查询 DataFrame
    model_select = model.lower()
    result = df[
        (df['Backend'] == backend) & 
        (df['Model_lower'] == model_select) & 
        (df['TP'] == tp) &
        (df['Device'] == device) & 
        (df['Clients'] == clients) &
        (df['Input-Output'] == input_output)
    ]
    return result


# 回调函数更新 TTFT 折线图 及 QPS 柱状图
@app.callback(
    [Output('ITL-graph', 'figure'), Output('ttft-graph', 'figure'),Output('qps-bar-chart', 'figure')],
    [Input('backend-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('dtype-dropdown', 'value'),
     Input('tp-dropdown', 'value'),
     Input('input-output-dropdown', 'value'),
     Input('client-dropdown', 'value'),
     Input('device-dropdown', 'value'),
     Input('upload-status', 'data')
     ]
)
def update_ttft_itl_graph(selected_backend, selected_model, select_dtype, selected_tp, input_output, select_clients, select_device, upload_status):
    if selected_backend is None:
        filtered_df = get_filtered_data(selected_model, select_dtype, selected_tp, input_output, select_clients, select_device)
        print(filtered_df)
        
        backend_order = ['vllm', 'trt', 'sglang', 'lightllm', 'lmdeploy', 'ppl']
        # 确保按照指定顺序排列
        filtered_df['Backend'] = pd.Categorical(filtered_df['Backend'], categories=backend_order, ordered=True)
        filtered_df = filtered_df.sort_values('Backend')  # 根据 Backend 的顺序进行排序
        # 使用 Plotly 生成折线图，确保横轴分类正确排序
        ttft_fig = go.Figure()
        ttft_fig.add_trace(go.Scatter(x=filtered_df['Backend'], y=filtered_df['P99 TTFT (s)'], mode='lines+markers', name='P99 TTFT (s)', line_shape='linear'))
        ttft_fig.add_trace(go.Scatter(x=filtered_df['Backend'], y=filtered_df['P50 TTFT (s)'], mode='lines+markers', name='P50 TTFT (s)', line_shape='linear'))
        ttft_fig.add_trace(go.Scatter(x=filtered_df['Backend'], y=filtered_df['P25 TTFT (s)'], mode='lines+markers', name='P25 TTFT (s)', line_shape='linear'))

        ttft_fig.update_layout(
            title=f'TTFT for {selected_model} ',
            xaxis_title=None,
            yaxis_title='TTFT (s)',
            legend_title='Percentiles',
            height=350,  # 调整图表高度
            width=500,  # 设置图表宽度
            margin=dict(l=20, r=20, t=40, b=20),  # 调整图表外边距
            xaxis=dict(categoryorder='array', categoryarray=backend_order, tickfont=dict(size=14)),  # 确保顺序正确
        )
        
        # 使用 Plotly 生成折线图，确保横轴分类正确排序
        itl_fig = go.Figure()
        itl_fig.add_trace(go.Scatter(x=filtered_df['Backend'], y=filtered_df['P99 ITL (ms)'], mode='lines+markers', name='P99 ITL (ms)', line_shape='linear'))
        itl_fig.add_trace(go.Scatter(x=filtered_df['Backend'], y=filtered_df['P50 ITL (ms)'], mode='lines+markers', name='P50 ITL (ms)', line_shape='linear'))
        itl_fig.add_trace(go.Scatter(x=filtered_df['Backend'], y=filtered_df['P25 ITL (ms)'], mode='lines+markers', name='P25 ITL (ms)', line_shape='linear'))

        itl_fig.update_layout(
            title=f'ITL for {selected_model}',
            xaxis_title=None,
            yaxis_title='ITL (ms)',
            legend_title='Percentiles',
            height=350,  # 调整图表高度
            width=500,  # 设置图表宽度
            margin=dict(l=20, r=20, t=40, b=20),  # 调整图表外边距
            xaxis=dict(categoryorder='array', categoryarray=backend_order, tickfont=dict(size=14)),  # 确保顺序正确
        )

        # 生成QPS柱状图
        qps_fig = go.Figure()
        backend_color_map = {'vllm': 'yellow', 'trt':'black', 'sglang': 'orange','lightllm': 'blue', 'lmdeploy': 'green',  'ppl':'red'}
        for backend, color in backend_color_map.items():
            y_value = filtered_df.loc[filtered_df['Backend_lower'] == backend, 'Total QPS'].values
            y_value = y_value[0] if len(y_value) > 0 else 0
            qps_fig.add_trace(go.Bar(x=[backend], y=[y_value], name=backend, marker=dict(color=color, line=dict(color='black', width=1.5)),width=0.2))
        qps_fig.update_layout(
            title=f'QPS for {selected_model}',
            xaxis_title=None,
            yaxis_title='QPS (req/s)',
            legend_title='Backend',
            margin=dict(l=20, r=20, t=40, b=20),  # 调整图表外边距
            xaxis=dict(categoryorder='array', categoryarray=list(backend_color_map.keys()), tickfont=dict(size=14)),
            barmode='relative',  # 使用相对模式，保持柱子的相对位置
            height=350,
            width=500
        )
        return itl_fig, ttft_fig, qps_fig
    else:
        filtered_df = get_filtered_data_dtype(selected_backend, selected_model, selected_tp, select_device, select_clients, input_output)
        print(filtered_df)
        
        # 限定 dtype 为指定的选项
        dtype_order = ['bf16', 'w4a16', 'w8a16', 'w8a8']

        # 确保 dtype 的顺序正确
        filtered_df['Dtype'] = pd.Categorical(filtered_df['Dtype'], categories=dtype_order, ordered=True)
        filtered_df = filtered_df.sort_values('Dtype')
        
        # TTFT 折线图
        ttft_fig = go.Figure()
        ttft_fig.add_trace(go.Scatter(x=filtered_df['Dtype'], y=filtered_df['P99 TTFT (s)'], mode='lines+markers', name='P99 TTFT (s)', line_shape='linear'))
        ttft_fig.add_trace(go.Scatter(x=filtered_df['Dtype'], y=filtered_df['P50 TTFT (s)'], mode='lines+markers', name='P50 TTFT (s)', line_shape='linear'))
        ttft_fig.add_trace(go.Scatter(x=filtered_df['Dtype'], y=filtered_df['P25 TTFT (s)'], mode='lines+markers', name='P25 TTFT (s)', line_shape='linear'))

        ttft_fig.update_layout(
            title=f'TTFT for {selected_model} under {selected_backend} ',
            xaxis_title=None,
            yaxis_title='TTFT (s)',
            legend_title='Percentiles',
            height=350,  # 调整图表高度
            width=500,  # 设置图表宽度
            margin=dict(l=20, r=20, t=40, b=20),  # 调整图表外边距
            xaxis=dict(categoryorder='array', categoryarray=dtype_order, tickfont=dict(size=14)),  # 确保顺序正确
        )
        # ITL 折线图
        itl_fig = go.Figure()
        itl_fig.add_trace(go.Scatter(x=filtered_df['Dtype'], y=filtered_df['P99 ITL (ms)'], mode='lines+markers', name='P99 ITL (ms)', line_shape='linear'))
        itl_fig.add_trace(go.Scatter(x=filtered_df['Dtype'], y=filtered_df['P50 ITL (ms)'], mode='lines+markers', name='P50 ITL (ms)', line_shape='linear'))
        itl_fig.add_trace(go.Scatter(x=filtered_df['Dtype'], y=filtered_df['P25 ITL (ms)'], mode='lines+markers', name='P25 ITL (ms)', line_shape='linear'))
        
        itl_fig.update_layout(
            title=f'ITL for {selected_model} under {selected_backend} ',
            xaxis_title=None,
            yaxis_title='ITL (ms)',
            legend_title='Percentiles',
            height=350,  # 调整图表高度
            width=500,  # 设置图表宽度
            margin=dict(l=20, r=20, t=40, b=20),  # 调整图表外边距
            xaxis=dict(categoryorder='array', categoryarray=dtype_order, tickfont=dict(size=14)),  # 确保顺序正确
        )
        
        # 生成QPS柱状图
        qps_fig = go.Figure()
        dtype_color_map = {'bf16':'navy',  'w4a16': 'gold',  'w8a16': 'coral',  'w8a8': 'teal'}
        for dtype, color in dtype_color_map.items():
            y_value = filtered_df.loc[filtered_df['Dtype'] == dtype, 'Total QPS'].values
            y_value = y_value[0] if len(y_value) > 0 else 0
            qps_fig.add_trace(go.Bar(x=[dtype], y=[y_value], name=dtype, marker=dict(color=color, line=dict(color='black', width=1.5)),width=0.2))
        qps_fig.update_layout(
            title=f'QPS for {selected_model} under {selected_backend}',
            xaxis_title=None,
            yaxis_title='QPS (req/s)',
            legend_title='dtype',
            margin=dict(l=20, r=20, t=40, b=20),  # 调整图表外边距
            xaxis=dict(categoryorder='array', categoryarray=list(dtype_color_map.keys()), tickfont=dict(size=14)),
            barmode='relative',  # 使用相对模式，保持柱子的相对位置
            height=350,
            width=500
        )
        
        return itl_fig, ttft_fig, qps_fig


# 回调函数：根据所选的 Model 更新 TP 的选项，并设置默认值
@app.callback(
    [Output('tp-dropdown', 'options'),
     Output('tp-dropdown', 'value')],  # value 用于设置默认值
    Input('model-dropdown', 'value')
)
def update_tp_options(selected_model):
    if not selected_model:
        return [], None

    # 将所选的 model 转为小写（因为之前列名已经变成小写）
    selected_model = selected_model.lower()

    # 筛选出当前选定的 model 对应的 TP
    filtered_df = df[df['Model_lower'] == selected_model]

    # 获取唯一的 TP 值并构造成 Dropdown 选项
    tp_values = filtered_df['TP'].unique()
    tp_options = [{'label': tp, 'value': tp} for tp in tp_values]

    # 设置默认值为第一个 TP
    default_tp = tp_values[0] if len(tp_values) > 0 else None

    return tp_options, default_tp


@app.callback(
    [Output('device-dropdown', 'options'),
     Output('device-dropdown', 'value')],
    [Input('model-dropdown', 'value'),
     Input('tp-dropdown', 'value')],
    [State('device-dropdown', 'value')]  # 保持现有的选择状态
)
def update_device_options(selected_model, selected_tp, current_device):
    # 根据选中的 model 和 tp 过滤 device
    if selected_model and selected_tp:
        filtered_df = df[(df['Model_lower'] == selected_model.lower()) & (df['TP'] == selected_tp)]
        device_options = [{'label': device, 'value': device} for device in filtered_df['Device'].unique()]
        # 如果当前选择的 device 仍然在新的选项列表中，保持该选择
        if current_device in [opt['value'] for opt in device_options]:
            return device_options, current_device
        # 否则返回空值
        return device_options, None
    return [], None


@app.callback(
    [Output('dtype-dropdown', 'options'),
     Output('dtype-dropdown', 'value')],
    [Input('model-dropdown', 'value'),
     Input('tp-dropdown', 'value'),
     Input('device-dropdown', 'value')],
    [State('dtype-dropdown', 'value')]  # 保持现有的选择状态
)
def update_dtype_options(selected_model, selected_tp, selected_device, current_dtype):
    # 根据选中的 model, tp 和 device 过滤 dtype
    if selected_model and selected_tp and selected_device:
        filtered_df = df[(df['Model_lower'] == selected_model.lower()) & 
                         (df['TP'] == selected_tp) & 
                         (df['Device'] == selected_device)]
        dtype_options = [{'label': dtype, 'value': dtype} for dtype in filtered_df['Dtype'].unique()]
        # 如果当前选择的 dtype 仍然在新的选项列表中，保持该选择
        if current_dtype in [opt['value'] for opt in dtype_options]:
            return dtype_options, current_dtype
        return dtype_options, None
    return [], None


@app.callback(
    [Output('client-dropdown', 'options'),
     Output('client-dropdown', 'value')],
    [Input('model-dropdown', 'value'),
     Input('tp-dropdown', 'value'),
     Input('device-dropdown', 'value'),
     Input('dtype-dropdown', 'value')],
    [State('client-dropdown', 'value')]  # 保持现有的选择状态
)
def update_clients_options(selected_model, selected_tp, selected_device, selected_dtype, current_clients):
    # 根据选中的 model, tp, device 和 dtype 过滤 clients
    if selected_model and selected_tp and selected_device and selected_dtype:
        filtered_df = df[(df['Model_lower'] == selected_model.lower()) & 
                         (df['TP'] == selected_tp) & 
                         (df['Device'] == selected_device) & 
                         (df['Dtype'] == selected_dtype)]
        clients_options = [{'label': client, 'value': client} for client in filtered_df['Clients'].unique()]
        # 如果当前选择的 client 仍然在新的选项列表中，保持该选择
        if current_clients in [opt['value'] for opt in clients_options]:
            return clients_options, current_clients
        return clients_options, None
    return [], None


@app.callback(
    [Output('input-output-dropdown', 'options'),
     Output('input-output-dropdown', 'value')],
    [Input('model-dropdown', 'value'),
     Input('tp-dropdown', 'value'),
     Input('device-dropdown', 'value'),
     Input('dtype-dropdown', 'value'),
     Input('client-dropdown', 'value')],
    [State('input-output-dropdown', 'value')]  # 保持现有的选择状态
)
def update_io_options(selected_model, selected_tp, selected_device, selected_dtype, selected_clients, current_io):
    # 根据选中的 model, tp, device, dtype 和 clients 过滤 input-output
    if selected_model and selected_tp and selected_device and selected_dtype and selected_clients:
        filtered_df = df[(df['Model_lower'] == selected_model.lower()) & 
                         (df['TP'] == selected_tp) & 
                         (df['Device'] == selected_device) & 
                         (df['Dtype'] == selected_dtype) &
                         (df['Clients'] == selected_clients)]
        io_options = [{'label': io, 'value': io} for io in filtered_df['Input-Output'].unique()]
        # 如果当前选择的 input-output 仍然在新的选项列表中，保持该选择
        if current_io in [opt['value'] for opt in io_options]:
            return io_options, current_io
        return io_options, None
    return [], None



# 回调函数：更新静态表格下拉框选项
@app.callback(
    [Output('world-size-dropdown', 'options'),
     Output('batch-size-dropdown', 'options'),
     Output('devices-dropdown', 'options'),
     Output('input-len-dropdown', 'options'),
     Output('output-len-dropdown', 'options')],
    [Input('model-dropdown', 'value')]
)
def update_filter_options(selected_model):
    start_time3 = time.time()
    # print("selected_model is:",selected_model)
    if not selected_model:
        return [], [], [], []
    
    selected_model = selected_model.lower()
    filtered_df = combined_df[combined_df['test_model_lower'] == selected_model]
    # 将 unique 的计算结果缓存为 Series，减少对 DataFrame 的重复操作
    world_sizes = filtered_df['world_size'].dropna().unique()
    batch_sizes = filtered_df['batch_size'].dropna().unique()
    devices = filtered_df['devices'].dropna().unique()
    input_lens = filtered_df['input_len'].dropna().unique()
    output_lens = filtered_df['output_len'].dropna().unique()
     # 构建 options 列表
    world_size_options = [{'label': str(size), 'value': size} for size in world_sizes]
    batch_size_options = [{'label': str(size), 'value': size} for size in batch_sizes]
    devices_options = [{'label': str(size), 'value': size} for size in devices]
    input_len_options = [{'label': str(length), 'value': length} for length in input_lens]
    output_len_options = [{'label': str(length), 'value': length} for length in output_lens]

    end_time3 = time.time()
    print(f"execution3 time: {end_time3 - start_time3} seconds")
    return world_size_options, batch_size_options, devices_options, input_len_options, output_len_options



# 回调函数：更新静态表格数据
@app.callback(
    Output('md-table', 'data'),
    [Input('model-dropdown', 'value'),
     Input('world-size-dropdown', 'value'),
     Input('batch-size-dropdown', 'value'),
     Input('devices-dropdown', 'value'),
     Input('input-len-dropdown', 'value'),
     Input('output-len-dropdown', 'value')]
)
def update_table(selected_model, world_size, batch_size, device, input_len, output_len):
    # 过滤数据
    filtered_df = combined_df[(combined_df['test_model'].str.lower() == selected_model.lower()) &
                              (combined_df['world_size'] == world_size) &
                              (combined_df['batch_size'] == batch_size) &
                              (combined_df['devices'] == device) &
                              (combined_df['input_len'] == input_len) &
                              (combined_df['output_len'] == output_len)]

    if not filtered_df.empty:
        return filtered_df[display_columns].to_dict('records')
    else:
        # 如果没有找到数据，返回0的表格
        empty_data = {col: [0] for col in display_columns}
        empty_df = pd.DataFrame(empty_data)
        return empty_df.to_dict('records')


# 处理网页上传数据并更新数据
@app.callback(
     [Output('output-data-upload', 'children'),
      Output('confirm-dialog_failed', 'message'),
      Output('confirm-dialog_failed', 'displayed'),
      Output('confirm-dialog_successd', 'displayed'),
      Output('upload-status', 'data'),
      Output('upload-data', 'contents')], 
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')]
)
def upload_files(contents, filenames, last_modified):
    global df
    if contents is not None:
        # 初始化成功与失败的提示
        success = False
        failure_message = None
        
        # 定义保存文件的目录路径
        upload_directory = '/nvme/sangchengmeng/result_save3/uploaded_data/'
        
        # 如果多个文件同时上传，`contents` 和 `filenames` 都是列表
        if isinstance(contents, list) and isinstance(filenames, list):
            error_messages = []
            
            for content, filename in zip(contents, filenames):
                # 解析文件内容（base64编码）
                content_type, content_string = content.split(',')

                # 验证文件名格式
                is_valid, error_message = validate_filename(filename)
                if not is_valid:
                    error_messages.append(f'Error:{filename}命名错误: {error_message}')
                    continue  # 如果文件名不符合规范，跳过该文件并继续处理下一个文件

                # 检查目录是否存在，不存在则创建
                if not os.path.exists(upload_directory):
                    os.makedirs(upload_directory)

                # 文件保存路径
                save_path = os.path.join(upload_directory, filename)

                # 解码并保存文件
                decoded = base64.b64decode(content_string)
                with open(save_path, 'wb') as f:
                    f.write(decoded)

                # 解析文件并更新全局数据
                new_data = parse_file(save_path)
                new_data = pd.DataFrame([new_data])
                df = pd.concat([df, new_data], ignore_index=True)
                df['Model_lower'] = df['Model'].str.lower()
                df['Backend_lower'] = df['Backend'].str.lower()

            if error_messages:
                # 如果有错误文件，显示错误提示
                failure_message = "\n".join(error_messages)
                return None, failure_message, True, False, {'uploaded': False}, None
            else:
                # 如果全部文件都成功上传
                success = True
                return f'Successfully uploaded files: {", ".join(filenames)}', None, False, True, {'uploaded': True}, None
        
        else:
            # 单个文件处理（防止有时候只上传一个文件）
            content_type, content_string = contents.split(',')

            # 验证文件名格式
            is_valid, error_message = validate_filename(filenames)
            if not is_valid:
                return None, f'Error: {error_message}', True, False, {'uploaded': False}, None

            # 检查目录是否存在，不存在则创建
            if not os.path.exists(upload_directory):
                os.makedirs(upload_directory)

            # 文件保存路径
            save_path = os.path.join(upload_directory, filenames)

            # 解码并保存文件
            decoded = base64.b64decode(content_string)
            with open(save_path, 'wb') as f:
                f.write(decoded)

            # 解析文件并更新全局数据
            new_data = parse_file(save_path)
            new_data = pd.DataFrame([new_data])
            df = pd.concat([df, new_data], ignore_index=True)
            df['Model_lower'] = df['Model'].str.lower()
            df['Backend_lower'] = df['Backend'].str.lower()

            # 成功上传单个文件
            success = True
            return f'Successfully uploaded file: {filenames}', None, False, True, {'uploaded': True}, None
        
    # 没有上传任何文件的情况下
    return None, None, False, False, {'uploaded': False}, None



# 监听 ConfirmDialog 的确认按钮点击并刷新页面
@app.callback(
    Output('page-refresh', 'href'),  # 使用 href 属性进行刷新
    Input('confirm-dialog_successd', 'submit_n_clicks')  # 监听确认按钮点击事件
)
def refresh_page(submit_n_clicks):
    if submit_n_clicks:  # 如果点击了确认按钮
        return '/'  # 返回主页 URL，刷新页面
    return None  # 如果没有点击，不刷新

# 运行应用
if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(host="0.0.0.0", port=8000)
