import time
import os
import json
from io import StringIO
from dash import Dash, dcc, html, Input, Output
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import plotly.figure_factory as ff
import plotly.express as px
# from scipy.stats import gaussian_kde
from collections import Counter

import dash
import dash_bootstrap_components as dbc
import numpy as np

import base64
import datetime
import io

import pandas as pd

import requests
import plotly.express as px

from datetime import date

from web.pages.constant import mock_data

EXPLAINER = """This is the sample simulator version 01."""

dash.register_page(__name__, name='Simulator', title='AST-APM | Simulator')

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365)

simulator_form_data = {
    "input_data_source": "NTSB",
    "input_model_name": "LSTM_Predictor",
    "input_from_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
    "input_to_date": end_date.strftime("%Y-%m-%d %H:%M:%S")
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

# https://dash.plotly.com/dash-core-components/datepickerrange
date_range = dcc.DatePickerRange(
    id="date_range_picker_id",
    min_date_allowed=datetime.date(2020, 1, 1),
    max_date_allowed=datetime.date.today(),
    initial_visible_month=datetime.date.today(),
    start_date=start_date,
    end_date=end_date,
    display_format="YYYY-MM-DD",
    className="mb-6"
)

layout = dbc.Container(
    [
        html.H1("Data Analyst Simulator"),
        dcc.Markdown(EXPLAINER),
        dbc.Row([
            dbc.Col(source_selector, width=3),
            dbc.Col(model_type_selector, width=3),
            dbc.Col(date_range, width=4),
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
                dbc.Tab(label="Performance", tab_id="histogram"),
                dbc.Tab(label="Trend View", tab_id="bar"),
            ],
            id="tabs",
            active_tab="info",
        ),
        # Ref:
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
def wordCountDistribution():
    # https://plotly.com/python/distplot/
   # Example data (replace with your actual DataFrame)
    # data = {'narrative_word_count': [120, 150, 300, 250, 400, 500, 320, 180, 240, 360, 410]}
    # df = pd.DataFrame(data)

    df = px.data.tips()
    hist_data = [df['total_bill']]  # List of lists
    group_labels = ['Word Count']  # Name of the dataset

    # Create distplot with histogram and KDE
    fig = ff.create_distplot(hist_data, group_labels, bin_size=50, colors=['blue'], show_hist=True, show_rug=False)

    # Add mean and standard deviation as annotations
    mean = df['total_bill'].mean()
    std = df['total_bill'].std()
    fig.add_annotation(x=mean, y=0, text=f"μ = {mean:.2f}", showarrow=False, font=dict(size=12, color="black"))
    fig.add_annotation(x=mean + std, y=0, text=f"σ = {std:.2f}", showarrow=False, font=dict(size=12, color="black"))

    # Customize the layout
    fig.update_layout(
        title="Distribution of Word Counts (KDE)",
        xaxis_title="Number of Words",
        yaxis_title="Density",
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def totalUniqueWordCountDistribution():

    # Sample cleaned text DataFrame
    data = {
        'narrative': [
            "The aircraft experienced turbulence during takeoff.",
            "The pilot reported an engine failure.",
            "Flight attendants noticed smoke in the cabin.",
            "The aircraft was grounded due to technical issues.",
            "The engine was shut down after a fire warning.",
            "The pilot made an emergency landing.",
            "The plane was delayed due to weather conditions.",
            "A warning light came on in the cockpit.",
            "There was a loud noise during the descent.",
            "The pilot declared an emergency."
        ]
    }

    # Create DataFrame
    clean_df = pd.DataFrame(data)

    # Assuming you already have clean_df['narrative'] processed and split into words
    all_words = [word for text in clean_df['narrative'].values for word in text.split()]
    word_counts = Counter(all_words)

    # Extract the 50 most common words
    common_words = word_counts.most_common(50)
    labels, values = zip(*common_words)

    # Create a bar plot using Plotly Express
    fig = px.bar(x=labels, y=values, title="Top 50 Most Frequent Words", labels={'x': 'Words', 'y': 'Count'})

    # Customize layout
    fig.update_layout(
        xaxis_title="Words",
        yaxis_title="Frequency",
        xaxis_tickangle=-90  # Rotate the x-axis labels by 90 degrees
    )

    return fig

def trendView():
    fig = go.Figure()

    df = px.data.stocks()
    fig = px.line(df, x="date", y=df.columns,
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
    
    return fig

# Callback Functions
@callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    
    if active_tab and data is not None:
        if active_tab == "info":
            data = {
                'category_column': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'C', 'A']
            }
            df = pd.DataFrame(data)

            # Count the occurrences of each category
            category_counts = df['category_column'].value_counts().reset_index()
            category_counts.columns = ['category', 'count']

            # Create a horizontal countplot using Plotly Express
            fig_count = px.bar(category_counts, x='count', y='category', orientation='h', 
                        title="Horizontal Countplot using Plotly")
            return html.Div(
                [
                    dbc.Row(
                        dbc.Col(dcc.Graph(figure=fig_count), width=12),
                        style={'margin-bottom': '25px'}
                    ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=wordCountDistribution()), width=6),
                            dbc.Col(dcc.Graph(figure=totalUniqueWordCountDistribution()), width=6)  
                        ]                      
                    )
                ]
            )
        elif active_tab == "info_evaluation":

            print(data)

            report_df = data['classification_report']

            print(report_df)

            report_df = pd.DataFrame(report_df).transpose()
            report_df.drop(["accuracy"], inplace=True)

            print(report_df)

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

            return dbc.Container(
                dbc.Row(
                    dbc.Col(dcc.Graph(figure=fig), width=12),
                )
            )
        elif active_tab == "loss_accuracy":

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

            return dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_loss), width=6),
                    dbc.Col(dcc.Graph(figure=fig_accuracy), width=6),
                ]
            )
        elif active_tab == "histogram":
            
            conf_matrix = data['conf_matrix']
            labels = data['labels']
            # predicted_labels = data['predicted_labels']


            # lb_size = len(labels)

            # new_conf_matrix = np.zeros((lb_size, lb_size), dtype=int)
            # new_conf_matrix[:4, :4] = conf_matrix

            # print(conf_matrix)

            # print(new_conf_matrix)

            fig = go.Figure()

            # https://plotly.com/python/annotated-heatmap/

            # df = px.data.medals_wide(indexed=True)
            # fig = px.imshow(df, text_auto=True)

            # fig = ff.create_annotated_heatmap(conf_matrix, x=predicted_labels, y=labels, colorscale='Blues', aspect=True)
            fig = px.imshow(conf_matrix, text_auto=True, color_continuous_scale='Blues', aspect="auto")

            fig.update_layout(
                title='Confusion Matrix',
                xaxis=dict(title='Predicted Label'),
                yaxis=dict(title='True Label')
            )

            return dbc.Row(
                [
                    dcc.Graph(figure=fig)
                    # dbc.Col(dcc.Graph(figure=data["hist_1"]), width=6),
                    # dbc.Col(dcc.Graph(figure=data["hist_2"]), width=6),
                ]
            )
        elif active_tab == "bar":
            return html.Div([
                html.H4('Interactive color selection with simple Dash example'),
                dcc.Graph(figure=trendView()),
            ])

    return "No tab selected"


@callback(Output("store", "data"), [Input("buttonSimulator", "n_clicks")])
def do_simulate(n):
    """
    This callback generates three simple graphs from random data.
    """

    if not n:
       
        # default Data
        return mock_data

    # simulate expensive graph generation process
    time.sleep(2)

    model_report = train_model()

    return model_report

# https://dash.plotly.com/dash-core-components/dropdown
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

# https://dash.plotly.com/dash-core-components/datepickerrange
@callback(
    [Input('date_range_picker', 'start_date'),
     Input('date_range_picker', 'end_date')]
)
def change_date_range(start_date, end_date):
    if start_date is not None and end_date is not None:

        start_date_object = date.fromisoformat(start_date)
        start_date = start_date_object.strftime("%Y-%m-%d %H:%M:%S")

        end_date_object = date.fromisoformat(end_date)
        end_date = end_date_object.strftime("%Y-%m-%d %H:%M:%S")

        simulator_form_data["input_from_date"] = start_date
        simulator_form_data["input_to_date"] = end_date

        print(start_date, end_date)

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