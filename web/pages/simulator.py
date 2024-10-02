import time
from dash import dcc, html, Input, Output
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go

import plotly.figure_factory as ff
import plotly.express as px

import dash
import dash_bootstrap_components as dbc
import numpy as np

import datetime

import pandas as pd

import requests
import plotly.express as px

from datetime import date

from web.pages.constant import mock_data

dash.register_page(__name__, name='Simulator', title='AST-APM | Simulator')

end_date = date(2023, 12, 31)
start_date = end_date - datetime.timedelta(days=365*1)

simulator_form_data = {
    "input_data_source": "ASRS",
    "input_model_name": "LSTM_Predictor",
    "input_from_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
    "input_to_date": end_date.strftime("%Y-%m-%d %H:%M:%S"),
    "input_model_ls_version": 2
}

source_selector = dbc.Select(
    id="source_selector_id",
    options=[
        {"label": "ASRS", "value": "asrs"},
        {"label": "NTSB", "value": "ntsb"},
        {"label": "Combined", "value": "asrs_ntsb"},
    ],
    value="asrs",
    className="mb-2"
)

model_type_selector = dbc.Select(
    id="model_name_selector_id",
    options=[
        {"label": "HF Predictor", "value": "LSTM_Predictor"},
        # {"label": "HF Taxonomy Classifier", "value": "LS_Classifier"},
    ],
    value="LSTM_Predictor",
    className="mb-2"
)

# code adapted from (DatePickerRange | Dash for Python Documentation | Plotly, n.d.)
date_range = dcc.DatePickerRange(
    id="date_range_picker_id",
    min_date_allowed=datetime.date(2000, 1, 1),
    max_date_allowed=end_date,
    initial_visible_month=start_date,
    start_date=start_date,
    end_date=end_date,
    display_format="YYYY-MM-DD",
    className="mb-6"
)
# end of adapted code

model_ls_version_selector = dbc.Select(
    id="model_ls_selector_id",
    options=[
        {"label": "ASRS", "value": "asrs"},
        {"label": "NTSB", "value": "ntsb"},
        {"label": "Combined", "value": "asrs_ntsb"},
    ],
    value="asrs",
    className="mb-2"
)

layout = dbc.Container(
    [
        html.H1("Data Analyst Simulator (.v1)"),
        dcc.Markdown(""),
        dbc.Row([
            dbc.Col(source_selector, width=2),
            dbc.Col(model_type_selector, width=2),
            dbc.Col(date_range, width=4),
            dbc.Col(model_ls_version_selector, width=2),
            dbc.Col(dbc.Button(
                "Start Simulator",
                color="primary",
                id="buttonSimulator",
                className="mb-2",
            ), width=2)
        ]),
        dbc.Tabs(
            [
                dbc.Tab(label="Summary", tab_id="info"),
                dbc.Tab(label="Evaluation Overview", tab_id="info_evaluation"),
                dbc.Tab(label="Loss/Accuracy", tab_id="loss_accuracy"),
                dbc.Tab(label="Performance", tab_id="performance"),
                dbc.Tab(label="Trend View", tab_id="trend_view"),
            ],
            id="tabs",
            active_tab="info",
        ),
        dbc.Spinner(
            [
                dcc.Store(id="store"),
                html.Div(id="tab-content", className="p-4"),
            ],
            delay_show=100,
        ),
    ]
)

# ------- FIG Functions ---------------
def labelCountDistribution(data):

    value_count = data['train_label_value_count']
    df = pd.DataFrame(value_count)

    fig_count = px.bar(df, x='count', y='HFACS_Category_Value_Predict', orientation='h', 
        title="HFACS Taxonomy Label Distribution")
    
    return fig_count

def wordCountDistribution(data):
    narrative_word_count = data['train_narrative_word_count']

    hist_data = [narrative_word_count]
    group_labels = ['Word Count']

    # code adapted from (Distplots, n.d.)
    fig = ff.create_distplot(hist_data, group_labels, bin_size=50, colors=['blue'], show_hist=True, show_rug=False)

    mean = np.mean(narrative_word_count)
    std = np.std(narrative_word_count)

    # Customize the layout
    fig.update_layout(
        title="Distribution of Word Counts (KDE)",
        xaxis_title="Number of Words",
        yaxis_title="Density",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    # end of adapted code

    return fig

def totalUniqueWordCountDistribution(data):
    
    train_word_count = data['train_word_count']

    labels = train_word_count['labels']
    values = train_word_count['values']

    fig = px.bar(x=labels, y=values, title="Top 50 Most Frequent Words", labels={'x': 'Words', 'y': 'Count'})

    # Customize layout
    fig.update_layout(
        xaxis_title="Words",
        yaxis_title="Frequency",
        xaxis_tickangle=-90
    )

    return fig

def classification_report_overview(data):
    report_df = data['classification_report']

    report_df = pd.DataFrame(report_df).transpose()
    report_df.drop(["accuracy"], inplace=True)

    fig = go.Figure()

    # Add Precision, Recall, and F1-Score bars for each class
    fig.add_trace(go.Bar(
        x=report_df.index,
        y=report_df['precision'],
        name='Precision',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=report_df.index,
        y=report_df['recall'],
        name='Recall',
        marker_color='orange'
    ))

    fig.add_trace(go.Bar(
        x=report_df.index,
        y=report_df['f1-score'],
        name='F1-Score',
        marker_color='green'
    ))

    # Customize layout
    fig.update_layout(
        title='Classification Report',
        xaxis=dict(title='Class'),
        yaxis=dict(title='Score'),
        barmode='group'
    )

    return fig

def lossView(data):
    fig_loss = go.Figure()

    train_loss = data['train_loss']
    test_loss = data['test_loss']

    fig_loss.add_trace(go.Scatter(x=list(range(len(train_loss))), y=train_loss, mode='lines', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(x=list(range(len(test_loss))), y=test_loss, mode='lines', name='Validation Loss'))
    fig_loss.update_layout(
        title='Model Loss',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss'),
        showlegend=True
    )
    return fig_loss

def accuracyView(data):
    fig_accuracy = go.Figure()

    train_accuracy = data['train_accuracy']
    test_accuracy = data['test_accuracy']

    fig_accuracy.add_trace(go.Scatter(x=list(range(len(train_accuracy))), y=train_accuracy, mode='lines', name='Train Loss'))
    fig_accuracy.add_trace(go.Scatter(x=list(range(len(test_accuracy))), y=test_accuracy, mode='lines', name='Validation Loss'))
    fig_accuracy.update_layout(
        title='Model Accuracy',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss'),
        showlegend=True
    )
    return fig_accuracy

def trendView(data):
    fig = go.Figure()

    predict_data = data['sample_predict_view']

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(predict_data, orient='index')

    # Convert index to datetime format
    df.index = pd.to_datetime(df.index)  

    df = df.reset_index().rename(columns={'index': 'date'}) 

    # code adapted from (Range, n.d.)
    fig = px.line(df, x="date", y=df.columns[1:],
        hover_data={"date": "|%B %d, %Y"},
        title='custom tick labels')
    
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y",
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ))
    # end of adapted code
    
    return fig

# Callback Functions
@callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    
    if active_tab and data is not None:
        if active_tab == "info":

            return html.Div(
                [
                    dbc.Row(
                        dbc.Col(dcc.Graph(figure=labelCountDistribution(data)), width=12),
                        style={'margin-bottom': '25px'}
                    ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=wordCountDistribution(data)), width=6),
                            dbc.Col(dcc.Graph(figure=totalUniqueWordCountDistribution(data)), width=6)  
                        ]                      
                    )
                ]
            )
        elif active_tab == "info_evaluation":

            return dbc.Container(
                dbc.Row(
                    dbc.Col(dcc.Graph(figure=classification_report_overview(data)), width=12),
                )
            )
        elif active_tab == "loss_accuracy":

            return dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=lossView(data)), width=6),
                    dbc.Col(dcc.Graph(figure=accuracyView(data)), width=6),
                ]
            )
        elif active_tab == "performance":
            
            conf_matrix = data['conf_matrix']

            print(conf_matrix)
            labels = data['labels']

            lb_size = len(labels)

            num_columns = len(conf_matrix[0])

            fig = go.Figure()

            print(lb_size, num_columns)
            # code adapted from (Annotated, n.d.)
            if lb_size == num_columns:
                fig = px.imshow(conf_matrix, x=labels, y=labels, text_auto=True, color_continuous_scale='Blues', aspect="auto")
            else:
                fig = px.imshow(conf_matrix, text_auto=True, color_continuous_scale='Blues', aspect="auto")
            # end of adapted code

            fig.update_layout(
                title='Confusion Matrix',
                xaxis=dict(title='Predicted Label'),
                yaxis=dict(title='True Label')
            )

            return dbc.Row(
                [
                    dcc.Graph(figure=fig)
                ]
            )
        elif active_tab == "trend_view":
            return html.Div([
                html.H4('Interactive color selection with simple Dash example'),
                dcc.Graph(figure=trendView(data)),
            ])

    return "No tab selected"


@callback(Output("store", "data"), [Input("buttonSimulator", "n_clicks")])
def do_simulate(n):
    if not n:
       
        # default Data
        return mock_data

    # simulate expensive graph generation process
    time.sleep(2)

    model_report = train_model()

    return model_report

# code adapted from (Dropdown | Dash for Python Documentation | Plotly, n.d.)
@callback(
    Input("source_selector_id", "value"))
def display_color(value):
    print("data_source", value)

    simulator_form_data["input_data_source"] = value

@callback(
    Input("model_name_selector_id", "value"))
def change_model_name(value):
    print("model_name", value)
    
    simulator_form_data["input_model_name"] = value
# end of adapted code

# code adapted from (DatePickerRange | Dash for Python Documentation | Plotly, n.d.)
@callback(
    [Input('date_range_picker_id', 'start_date'),
     Input('date_range_picker_id', 'end_date')]
)
def change_date_range(start_date, end_date):

    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date = start_date_object.strftime("%Y-%m-%d %H:%M:%S")
        simulator_form_data["input_from_date"] = start_date
        print("Start date=", start_date)

    if end_date is not None:

        end_date_object = date.fromisoformat(end_date)
        end_date = end_date_object.strftime("%Y-%m-%d %H:%M:%S")
        simulator_form_data["input_to_date"] = end_date
        print("End Date=", end_date)
# end of adapted code

is_mock = False
# ------ API Actions ------------------
def train_model():
    
    print(simulator_form_data)

    if is_mock is False:
       
        response = requests.post(f"http://127.0.0.1:8000//ml-models/train", json=simulator_form_data)

        if response.status_code == 201:
            data = response.json()

            # df = pd.DataFrame(data)

            return data
        else:
            return None

    return mock_data

# TODO (DatePickerRange | Dash for Python Documentation | Plotly, n.d.)=https://dash.plotly.com/dash-core-components/datepickerrange
# TODO (Dropdown | Dash for Python Documentation | Plotly, n.d.)=https://dash.plotly.com/dash-core-components/dropdown
# https://plotly.com/python/annotated-heatmap/
# https://plotly.com/python/range-slider/
# https://plotly.com/python/#ai_ml
# https://plotly.com/python/distplot/