import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home', title='AST-APM | Home')

card_01 = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Dashboard", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Checkout Analytics", color="primary", href="/analytics?data_source=asrs&unique_id=20240613175806"),
            ]
        ),
    ]
)

card_02 = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("NTSB", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Checkout Analytics", color="primary", href="/analytics"),
            ]
        ),
    ]
)

card_03 = dbc.Card(
    [
        # dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
        dbc.CardBody(
            [
                html.H4("NTSB", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Checkout Analytics", color="primary", href="/analytics"),
            ]
        ),
    ]
)

card_04 = dbc.Card(
    [
        # dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
        dbc.CardBody(
            [
                html.H4("NTSB", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Checkout Analytics", color="primary", href="/analytics"),
            ]
        ),
    ]
)

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([
            html.H3(['Welcome!']),
            html.P([html.B(['Aviation Safety Trends Analysis and Predictive Dashboard Overview'])], className='par')
        ], width=12, className='row-titles')
    ]),

    dbc.Row(
        [
            dbc.Col([], width = 2),
            dbc.Col(card_01, width=4),
            dbc.Col(card_02, width=4),
            dbc.Col([], width = 2)
        ]
    ),

    dbc.Row(
        [
            dbc.Col([], width = 2),
            dbc.Col(card_03, width=4),
            dbc.Col(card_04, width=4),
            dbc.Col([], width = 2)
        ]
    ),

    # Guidelines
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            html.P([html.B('1) Dashboard overview'),html.Br(),
                    'Provides an overview of the overall data based on the data sources or a hybrid/mixed approach.'], className='guide'),
            html.P([html.B('2) Aviation Safety Predictive Simulator'),html.Br(),
                    'The available tools can simulate machine learning models based on the data sources, models, and hyperparameters.'], className='guide'),
            html.P([html.B('3) Data Query tool'),html.Br(),
                    'Query the specific data set and analysis.'
                    # 'The seasonality component of the model can be excluded by leaving all right-hand parameters to 0.',html.Br(),
                    ], className='guide'),
            html.P([html.B('4) Integrate new Data Source'),html.Br(),
                    'Integrate new data source to this system',html.Br()], className='guide')
        ], width = 8),
        dbc.Col([], width = 2)
    ])
])