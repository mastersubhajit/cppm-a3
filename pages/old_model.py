import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import joblib

dash.register_page(__name__, path="/old-model", name="Old Model")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "cppm_a1_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "cppm_a1_scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Cars.csv")

# Load
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(DATA_PATH)

# Required features
features = ["year", "max_power", "mileage"]

# Styles (Neumorphism)
NEUMORPHISM_STYLE = {"background": "#fff", "minHeight": "100vh", "padding": "40px", "fontFamily": "Inter, sans-serif", "color": "#2c3e50"}
CARD_STYLE = {"background": "#e0e5ec","boxShadow": "9px 9px 16px #a3b1c6, -9px -9px 16px #ffffff","borderRadius": "20px","padding": "28px","margin": "0 auto","maxWidth": "760px"}
ROW_STYLE = {"display": "grid","gridTemplateColumns": "1fr 1fr 1fr","gap": "16px","marginTop": "12px","alignItems": "center"}
LABEL_STYLE = {"fontSize": "14px","marginBottom": "6px","display": "block", "textAlign": "center"}
INPUT_STYLE = {"textAlign": "center","background": "#e0e5ec","boxShadow": "inset 5px 5px 10px #a3b1c6, inset -5px -5px 10px #ffffff","border": "none","borderRadius": "12px","padding": "12px","width": "200px","outline": "none","fontSize": "14px"}
BUTTON_STYLE = {"background": "#e0e5ec","boxShadow": "5px 5px 10px #a3b1c6, -5px -5px 10px #ffffff","border": "none","borderRadius": "12px","padding": "14px 24px","cursor": "pointer","fontWeight": "600","width": "200px","marginTop": "18px","display": "block","margin": "20px auto","textAlign": "center"}
OUTPUT_STYLE = {"background": "#e0e5ec","boxShadow": "inset 6px 6px 12px #a3b1c6, inset -6px -6px 12px #ffffff","borderRadius": "15px","padding": "18px","marginTop": "20px","textAlign": "center","fontSize": "20px","fontWeight": "700"}

# --- Layout ---
layout = html.Div(style=NEUMORPHISM_STYLE, children=[
    html.H1("Old Model Predictor", style={"textAlign": "center", "marginBottom": "24px"}),

    html.Div(style=CARD_STYLE, children=[
        html.Div(style=ROW_STYLE, children=[
            html.Div(children=[html.Label("Year of Manufacture", style=LABEL_STYLE), dcc.Input(id="old-year", type="number", value=2019, step=1, style=INPUT_STYLE)]),
            html.Div(children=[html.Label("Max Power (bhp)", style=LABEL_STYLE), dcc.Input(id="old-max_power", type="text", value=91, step=1, style=INPUT_STYLE)]),
            html.Div(children=[html.Label("Mileage (kmpl)", style=LABEL_STYLE), dcc.Input(id="old-mileage", type="text", value=13.3, step=0.1, style=INPUT_STYLE)]),
        ]),
        html.Button("Predict Price", id="old-predict-button", n_clicks=0, style=BUTTON_STYLE),
        html.Div(id="old-prediction-output", style=OUTPUT_STYLE),
    ]),
])

# --- Callback ---
@dash.callback(
    Output("old-prediction-output", "children"),
    Input("old-predict-button", "n_clicks"),
    State("old-year", "value"),
    State("old-max_power", "value"),
    State("old-mileage", "value"),
)
def predict_price_old(n_clicks, year, max_power, mileage):
    if n_clicks <= 0:
        return "Enter details and click Predict."
    year = int(year)
    max_power = float(max_power)
    mileage = float(mileage)
    X_input = np.array([[year, max_power, mileage]])
    X_scaled = scaler.transform(X_input)
    log_pred = model.predict(X_scaled)[0]
    price = float(np.exp(log_pred))
    return f"Estimated Selling Price: ₹{price:,.0f}"
