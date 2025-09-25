import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import os

# ===== Register page with Dash Pages =====
dash.register_page(
    __name__,
    path="/visualization",
    name="Visualization",
    title="Statistics & Graphs"
)

# ===== Load Dataset =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Cars.csv")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

vehicle_df = pd.read_csv(DATA_PATH)

# ===== Hero Section =====
def hero_section(title, subtitle):
    return dbc.Container(
        [
            html.H1(title, className="display-4 text-center mb-2"),
            html.P(subtitle, className="lead text-center text-muted"),
            html.Hr(),
        ],
        fluid=True,
        className="py-3",
    )

# ===== Layout for Visualization Page =====
layout = dbc.Container([
    hero_section("Statistics & Graphs", "Explore dataset insights"),

    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Year Distribution"),
                dcc.Graph(
                    figure=px.histogram(vehicle_df, x="year", nbins=20, title="Car Year Distribution")
                )
            ]), className="shadow-sm mb-3")
        ], width=6),

        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Fuel Type Count"),
                dcc.Graph(
                    figure=px.pie(vehicle_df, names="fuel", title="Fuel Type Share")
                )
            ]), className="shadow-sm mb-3")
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Price vs Mileage"),
                dcc.Graph(
                    figure=px.scatter(vehicle_df, x="mileage", y="selling_price", color="fuel",
                                      title="Price vs Mileage")
                )
            ]), className="shadow-sm mb-3")
        ], width=12),
    ])
], fluid=True)
