import dash
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from dash import dcc, html

dash.register_page(__name__, path='/', name='Dashboard', title='AST-APM | Dashboard')

 # code adapted from (Range, n.d.)
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
# end of adapted code

layout =  dbc.Container([

    dbc.Row([
        dbc.Col([html.H3(['Human Factors Trend Overview'])], width=12, className='row-titles')
    ]),

   dbc.Row(
        [
            # dbc.Col([], width = 2),
            dbc.Col(dcc.Graph(figure=fig))
        ]
    )
])