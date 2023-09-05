import pandas as pd
import numpy as np
import mplfinance as mplf

import base64
import datetime
import io
import time

import dash
from dash import dcc, html, dash_table


import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

import pyarrow.dataset as ds

SIDESTYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "29rem",
    "padding": "2rem 1rem",
    "background-color": "#222222",
}


CONTSTYLE = {
    "margin-left": "15rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


# Инициализация сервера
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def get_options_exh():
    dict_list = []
    list_stocks = [4, 5, 6, 7, 8, 9]
    for i in list_stocks:
        dict_list.append({'label': f'Эксгаустер {i}', 'value': i})

    return dict_list

def get_options_columns():
    dict_list = []
    list_stocks = ['ТОК РОТОРА 1', 'ТОК РОТОРА2', 'ТОК РОТОРА 2', 'ТОК СТАТОРА',
                   'ДАВЛЕНИЕ МАСЛА В СИСТЕМЕ', 'ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 1',
                   'ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 2', 'ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 3',
                   'ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 4', 'ТЕМПЕРАТУРА МАСЛА В СИСТЕМЕ',
                   'ТЕМПЕРАТУРА МАСЛА В МАСЛОБЛОКЕ', 'ВИБРАЦИЯ НА ОПОРЕ 1', 'ВИБРАЦИЯ НА ОПОРЕ 2',
                   'ВИБРАЦИЯ НА ОПОРЕ 3', 'ВИБРАЦИЯ НА ОПОРЕ 3. ПРОДОЛЬНАЯ.', 'ВИБРАЦИЯ НА ОПОРЕ 4',
                   'ВИБРАЦИЯ НА ОПОРЕ 4. ПРОДОЛЬНАЯ.'
                   ]
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})

    return dict_list

def get_exhauster(dat_inf='X_train', exh='', col=''):
    hash_data = ds.dataset(f"data/{dat_inf}.parquet", format="parquet")
    columns = ['DT']

    for i in hash_data.schema.names:
        if i.find(f'ЭКСГАУСТЕР {exh}. {col}') != -1:
            columns.append(i)

    return hash_data.to_table(columns=columns).to_pandas()

def get_technical(dat_inf='y_train', exh='', col=''):
    hash_data = ds.dataset(f"data/{dat_inf}.parquet", format="parquet")
    columns = ['DT']
    full_col = f'Y_ЭКСГАУСТЕР А/М №{exh}_{col}'

    for i in hash_data.schema.names:
        if i.find(f'{full_col}') != -1:
            columns.append(i)

    df = hash_data.to_table(columns=columns)

    columns[0] = 'TIME'
    arr = df.rename_columns(columns).to_pandas().to_numpy()

    new_array = [[arr[0][0], arr[0][0], arr[0][1]]]

    for i in range(1, len(arr)):
        if new_array[-1][2] != arr[i][1]:
            new_array.append([arr[i][0], arr[i][0], arr[i][1]])
        else:
            new_array[-1][1] = arr[i][0]
    df = list(filter(lambda component: component[2] > 0, new_array))

    return df


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'parquet' in filename:
            hash_data = ds.dataset(io.BytesIO(decoded), format="parquet")
            columns = []

            for i in hash_data.schema.names:
                columns.append(i)

            df = hash_data.to_table(columns=columns).to_pandas()
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5('Ваши данные:'),
        html.H5(filename),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])







app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(
        [
            html.H2("ИИСправность", className="display-4", style={'color': 'white'}),
            html.Hr(style={'color': 'white'}),
            dbc.Nav(
                [
                    dbc.NavLink("Тестирование моделей", href="/page3", active="exact"),
                    dbc.NavLink("Вывод графиков по работе датчиков", href="/page1", active="exact"),
                    dbc.NavLink("Существующие поломки", href="/page2", active="exact")
                ],
                vertical=True,
                pills=True),
        ],
        style=SIDESTYLE,
    ),
    html.Div(id="page-content", children=[], style=CONTSTYLE)
])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")])
def pagecontent(pathname):
    if pathname == "/page1":
        return [

            html.Div(
                children=[
                    html.Div(className='row',
                             children=[
                                 html.Div(className='four columns div-user-controls',
                                          children=[
                                              html.H2('''Визуализация рядов'''),
                                              html.P('''Выберете один эксгаустер'''),
                                              html.Div(
                                                  className='div-for-dropdown',
                                                  children=[
                                                      dcc.Dropdown(
                                                          id='get_options_exh',
                                                          options=get_options_exh(),
                                                          multi=False,
                                                          style={'backgroundColor': '#1E1E1E'},
                                                          className='get_options_exh'
                                                      ),
                                                  ],
                                                  style={'color': '#1E1E1E'}),
                                              html.P('''Выберете датчик эксгаустера'''),
                                              html.Div(
                                                  className='div-for-dropdown',
                                                  children=[
                                                      dcc.Dropdown(
                                                          id='get_options_columns',
                                                          options=get_options_columns(),
                                                          multi=False,
                                                          style={'backgroundColor': '#1E1E1E'},
                                                          className='get_options_columns'
                                                      ),
                                                  ],
                                                  style={'color': '#1E1E1E'})
                                          ]
                                          ),
                                 html.Div(className='eight columns div-for-charts bg-grey',
                                          children=[
                                              dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True)
                                          ])
                             ])
                ]
            )
                ]

    elif pathname == "/page2":
        return [

            html.Div(
                children=[
                    html.Div(className='row',
                             children=[
                                 html.Div(className='four columns div-user-controls',
                                          children=[
                                              html.H2('''Просмотр времени поломок'''),
                                              html.P('''Выберете один эксгаустер'''),
                                              html.Div(
                                                  className='div-for-dropdown',
                                                  children=[
                                                      dcc.Dropdown(
                                                          id='get_options_exh_2',
                                                          options=get_options_exh(),
                                                          value=4,
                                                          multi=False,
                                                          style={'backgroundColor': '#1E1E1E'},
                                                          className='get_options_exh_2'
                                                      ),
                                                  ],
                                                  style={'color': '#1E1E1E'}),
                                              html.P('''Выберете тех. место'''),
                                              html.Div(
                                                  className='div-for-dropdown',
                                                  children=[
                                                      dcc.Dropdown(
                                                          id='get_options_columns_2',
                                                          multi=False,
                                                          style={'backgroundColor': '#1E1E1E'},
                                                          className='get_options_columns_2'
                                                      ),
                                                  ],
                                                  style={'color': '#1E1E1E'}),
                                              dcc.Loading(id="ls-loading-1",
                                                          children=[html.Div(id="ls-loading-output-1")],
                                                          type="default"),
                                          ]
                                          ),
                                 html.Div(className='eight columns div-for-charts bg-grey', id="table-mest"),
                             ])
                ]
            )
        ]
    elif pathname == "/page3":
        return [
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(className='row',
                                     children=[
                                                  html.Div(className='four columns div-user-controls',
                                                           children=[
                                                               html.H2('''Просмотр результатов моделирования'''),
                                                               html.Button("Выгрузка результатов моделирования", id="btn_csv"),
                                                               dcc.Download(id="download-dataframe-csv"),
                                                               html.P(
                                                                   '''
                                                                    Загрузите csv файл со значениями датчиков для которых требуется построить прогноз поломок
                                                                    '''
                                                                ),
                                                               dcc.Upload(
                                                                   id='upload-x',
                                                                   children=html.Div([
                                                                       'Drag and Drop or ',
                                                                       html.A('Select Files')
                                                                   ]),
                                                                   style={
                                                                       'width': '100%',
                                                                       'height': '60px',
                                                                       'lineHeight': '60px',
                                                                       'borderWidth': '1px',
                                                                       'borderStyle': 'dashed',
                                                                       'borderRadius': '5px',
                                                                       'textAlign': 'center',
                                                                       'margin': '10px'
                                                                   }
                                                               ),
                                                               html.Div(id='output-x-upload')
                                                           ]
                                                           ),
                                         ]
                                     )
                            ]
                    ),

                ],
                className='header')
            ]


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    df = get_exhauster(exh='4', col='ТОК РОТОРА 1')
    return dcc.send_data_frame(df.to_csv, "mydf.csv")

@app.callback(Output("ls-loading-output-1", "children"), [Input("get_options_columns_2", "value"), Input("get_options_exh_2", "value")])
def input_triggers_spinner(value, value_2):
    if value is None or value_2 is None:
        time.sleep(1)
    else:
        time.sleep(25)

    return ''

@app.callback(Output('output-x-upload', 'children'),
              Input('upload-x', 'contents'),
              State('upload-x', 'filename'))
def update_output(content, name):
    if content is not None:
        children = parse_contents(content, name)
        return children


@app.callback(Output("get_options_columns_2", "options"),
              [Input('get_options_exh_2', 'value')])
def get_options_columns_2(exh):
    hash_data = ds.dataset(f"data/y_train.parquet", format="parquet")
    columns = []

    for i in hash_data.schema.names:
        if i.find(f'Y_ЭКСГАУСТЕР А/М №{exh}') != -1:
            columns.append(i.replace(f'Y_ЭКСГАУСТЕР А/М №{exh}_', ''))

    return columns


# # Callback
@app.callback(Output('table-mest', 'children'),
              [Input('get_options_exh_2', 'value'), Input("get_options_columns_2", "value")])
def update_table(exhauster, name_column):
    dataframe = []

    if exhauster is None or name_column is None:
        arr = [[0,0,0]]
    else:
        arr = get_technical(exh=exhauster, col=name_column)

    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in ['Начало', 'Конец', 'Тип поломки']])] +

        # Body
        [html.Tr([
            html.Td(col) for col in i
        ]) for i in arr]
    )


# # Callback
@app.callback(Output('timeseries', 'figure'),
              [Input('get_options_exh', 'value'), Input('get_options_columns', 'value')])
def update_graph(exhauster, name_column):
    df_graph = []
    trace1 = []
    title = ''
    yaxis = {}

    if exhauster is None or name_column is None:
        df_graph = get_exhauster()
        title = 'Эксгаустер'

        trace1.append(go.Scatter(
            x=df_graph[::500].index,
            y=[],
            mode='lines',
            opacity=0.7,
            name='',
            textposition='bottom center')
        )

        yaxis = {
            'range': [1, 100]
        }

    else:
        df_graph = get_exhauster(exh=exhauster, col=name_column)
        title = f'Эксгаустер {exhauster}. {name_column}'

        trace1.append(go.Scatter(
            x=df_graph[::500].index,
            y=df_graph[::500][f'ЭКСГАУСТЕР {exhauster}. {name_column}'],
            mode='lines',
            opacity=0.7,
            name=name_column,
            textposition='bottom center')
        )

        yaxis = {
            'range': [df_graph[f'ЭКСГАУСТЕР {exhauster}. {name_column}'].min(), df_graph[f'ЭКСГАУСТЕР {exhauster}. {name_column}'].max()]
        }

    traces = [trace1]
    data = [val for sublist in traces for val in sublist]

    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': title, 'font': {'color': 'white'}, 'x': 0.5},
                  xaxis={'range': [df_graph.index.min(), df_graph.index.max()]},
                  yaxis=yaxis
              ),

              }

    return figure

if __name__ == '__main__':
    app.run_server(host='localhost', port=80)