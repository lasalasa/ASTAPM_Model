# code adapted from Gabri-Al (n.d.)
from dash import html
import dash_bootstrap_components as dbc
import dash

_nav = dbc.Container([
	dbc.Row([
        dbc.Col([
            html.Div([
                html.I(className="fa-solid fa-chart-simple fa-2x")],
		    className='logo')
        ], width = 4),
        dbc.Col([html.H1(['AST-APD'], className='app-brand')], width = 8)
	]),
	dbc.Row([
        dbc.Nav(
	        [
                dbc.NavLink("Dashboard", active='exact', href="/dashboard"),
                dbc.NavLink("Simulator", active='exact', href="/dashboard/simulator")
                # dbc.NavLink(page["name"], active='exact', href=page["path"]) for page in dash.page_registry.values()
            ],
	        vertical=True, pills=True, class_name='my-nav')
    ])
])
# end of adapted code