# Import packages
import dash
import os
from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, ALL
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

from web.pages.data import file_name_list
# from side_bar import sidebar

# Incorporate data
df_event = pd.read_csv('web/data_store/asrs_store/csv/20240613175806/Events.csv')

df =  df_event.copy()
column = 'Anomaly'
df[column] = df[column].str.split(';')
contributing_factors = df[column].explode().str.strip()
contributing_factors_summary = contributing_factors.value_counts()
top_contributing_factors = contributing_factors_summary.reset_index()

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]

dash.register_page(__name__, name='Query Tool', title='AST-APM | Query Tool')

# button_group = dbc.ButtonGroup(
#     # [dbc.Button("Left"), dbc.Button("Middle"), dbc.Button("Right")]
#     [dbc.Button(btn_name, 
#                 id={'type': 'dynamic-button', 'index': btn_name}, 
#                 n_clicks=0, 
#                 size="md",
#                 className="me-1") for btn_name in name_list],
#     id='data-table-btn-group'
# )

select = dbc.Select(
    id="select-table-option",
    options=[
        {"label": btn_name, "value": btn_name} for btn_name in file_name_list
        # {"label": "Option 2", "value": "2"},
        # {"label": "Disabled option", "value": "3", "disabled": True},
    ],
    value=file_name_list[0]
)

# App layout
layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3(['Your dataset'])], width=12, className='row-titles')
    ]),

    dbc.Row(
        [
            dbc.Col([], width = 2),
            dbc.Col(
                [
                    dbc.Row([
                        html.H1(['ASRS Data View'], className="text-primary text-center fs-3")
                    ]),

                    dbc.Row([
                        dbc.Col([
                            select
                        ], width=2)
                    ]),

                    # dbc.Row([
                    #     dbc.Checklist(options=[{"label": x, "value": x} for x in ['Miss Distance', 'Passengers Involved', 'Detected']],
                    #                             value=[], 
                    #                             inline=True, 
                    #                             id='controls-and-radio-item'),
                    # ]),
                    dbc.Row([
                        html.H2([""], id="table-lbl-name")
                        # html.Div('ASRS Data View', id="table-lbl-name", className="text-primary fs-3")
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="loading-table",
                                type="default",
                                children=[ dash_table.DataTable(data=[], page_size=10, style_table={'overflowX': 'auto'}, id='data-table')]
                            )
                        ], width=12)
                    ]),
                    # dbc.Row([
                    #     dbc.Col([
                    #         dcc.Graph(figure={}, id='controls-and-graph')
                    #     ], width=6)
                    # ])
                ], width=10
            )
        ]
    )
], fluid=True)

# View Table Data
@callback(
    Output(component_id='data-table', component_property='data'),
    Output(component_id='table-lbl-name', component_property='children'),
    Input(component_id='select-table-option', component_property='value'),
    # Input({'type': 'dynamic-button', 'index': ALL}, 'n_clicks')
)
def update_table(value):
    # print(col_chosen)
    # records = asrs_data[col_chosen].to_dict('records')
    # fig = px.histogram(top_contributing_factors.head(10), x='Anomaly', y='count', histfunc='avg')
    # dash_table.DataTable(data=top_contributing_factors.to_dict('records'), page_size=10, style_table={'overflowX': 'auto'})
    
    FOLDER_PATH_DATA_STORE = "data_store"
    DATA_STORE_TYPES = ["csv"]
    DATA_SOURCES = ["asrs", "ntsb"]

    data_type = DATA_STORE_TYPES[0]
    source_name = DATA_SOURCES[0]
    unique_id = "20240613175806"

    folder_path = os.path.join(FOLDER_PATH_DATA_STORE, f"{source_name}_store",data_type, unique_id)

    # triggered_id = ctx.triggered_id
    table_name = 'Aircraft 1.csv'
    if value != None:
        # df = pd.read_csv(os.path.join(folder_path, table_name), low_memory=False)
        # records = df.to_dict('records')
    # else:
        table_name =value

        # records = asrs_data[table_name].to_dict('records')
    
    df = pd.read_csv(os.path.join(folder_path, table_name), low_memory=False)
    records = df.to_dict('records')
    return records, table_name

# Add controls to build the interaction
# @callback(
#     Output(component_id='controls-and-graph', component_property='figure'),
#     Input(component_id='controls-and-radio-item', component_property='value')
# )
# def update_graph(col_chosen):
#     fig = px.histogram(top_contributing_factors.head(10), x='Anomaly', y='count', histfunc='avg')

#     return fig
