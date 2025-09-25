import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import os

import dash
from dash import Dash, html, dcc

# Initialize the Dash app with Bootstrap + Neumorphism CSS
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server

# --- Navbar ---
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/", className="nav-link-neumorphism")),
        # NEW A3 MODEL LINK
        dbc.NavItem(dbc.NavLink("A3 Model", href="/a3-model", className="nav-link-neumorphism")),
        dbc.NavItem(dbc.NavLink("A2 Model", href="/new-model", className="nav-link-neumorphism")),
        dbc.NavItem(dbc.NavLink("A1 Model", href="/old-model", className="nav-link-neumorphism")),
        dbc.NavItem(dbc.NavLink("Visualization", href="/visualization", className="nav-link-neumorphism")),
    ],
    brand="Car Price Predictor",
    brand_href="/",
    color="transparent",
    dark=False,
    className="mb-4 neumorphism-navbar"
)

# --- Layout ---
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    dash.page_container
])

# Run
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)