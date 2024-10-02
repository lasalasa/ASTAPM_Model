# code adapted from Gabri-Al (n.d.)
import dash
import dash_bootstrap_components as dbc
from dash import dcc

app = dash.Dash(
    __name__, 
    use_pages=True, 
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    # code adapted from (Redditinc, n.d.)
    requests_pathname_prefix='/dashboard/',
    # end of adapted code
	suppress_callback_exceptions=True, 
    prevent_initial_callbacks=True
)

from web.components.footer import _footer
from web.components.nav import _nav

# App Layout
app.layout = dbc.Container([
	
	dbc.Row([
        dbc.Col([_nav], width = 2),
        dbc.Col([
            dbc.Row([dash.page_container])
	    ], width = 10),
    ]),
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            dbc.Row([_footer])
	    ], width = 10),
    ]),
    dcc.Store(id='browser-memo', data=dict(), storage_type='session')
], fluid=True)

if __name__ == '__main__':
    app.run(debug=True)
# end of adapted code