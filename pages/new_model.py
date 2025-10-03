import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import joblib

dash.register_page(__name__, path="/new-model", name="A2 Model")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "cppm_a2_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "cppm_a2_scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Cars.csv")

# Load
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(DATA_PATH)

features = ["year", "max_power", "mileage"]

# --- Styles (reuse from old_model.py for consistency) ---
NEUMORPHISM_STYLE = {"background": "#fff", "minHeight": "100vh", "padding": "40px", "fontFamily": "Inter, sans-serif", "color": "#2c3e50"}
CARD_STYLE = {"background": "#e0e5ec","boxShadow": "9px 9px 16px #a3b1c6, -9px -9px 16px #ffffff","borderRadius": "20px","padding": "28px","margin": "0 auto","maxWidth": "760px"}
ROW_STYLE = {"display": "grid","gridTemplateColumns": "1fr 1fr 1fr","gap": "16px","marginTop": "12px","alignItems": "center"}
LABEL_STYLE = {"fontSize": "14px","marginBottom": "6px","display": "block", "textAlign": "center"}
INPUT_STYLE = {"textAlign": "center","background": "#e0e5ec","boxShadow": "inset 5px 5px 10px #a3b1c6, inset -5px -5px 10px #ffffff","border": "none","borderRadius": "12px","padding": "12px","width": "200px","outline": "none","fontSize": "14px", "textAlign": "center"}
BUTTON_STYLE = {"background": "#e0e5ec","boxShadow": "5px 5px 10px #a3b1c6, -5px -5px 10px #ffffff","border": "none","borderRadius": "12px","padding": "14px 24px","cursor": "pointer","fontWeight": "600","width": "200px","marginTop": "18px","display": "block","margin": "20px auto","textAlign": "center"}
OUTPUT_STYLE = {"background": "#e0e5ec","boxShadow": "inset 6px 6px 12px #a3b1c6, inset -6px -6px 12px #ffffff","borderRadius": "15px","padding": "18px","marginTop": "20px","textAlign": "center","fontSize": "20px","fontWeight": "700"}

# --- Modal Styles ---
MODAL_STYLE = {"position": "fixed", "top": "0", "left": "0", "width": "100%", "height": "100%",
               "backgroundColor": "rgba(0,0,0,0.4)", "display": "flex", "justifyContent": "center", "alignItems": "center"}
MODAL_CONTENT_STYLE = {"background": "#e0e5ec","boxShadow": "9px 9px 16px #a3b1c6, -9px -9px 16px #ffffff",
                       "borderRadius": "20px","padding": "24px","maxWidth": "600px","textAlign": "center"}
# Button inside modal
BUTTON_STYLE_MODAL = {"background": "#e0e5ec","boxShadow": "5px 5px 10px #a3b1c6, -5px -5px 10px #ffffff",
                      "border": "none","borderRadius": "12px","padding": "10px 20px","cursor": "pointer",
                      "fontWeight": "600","marginTop": "20px"}
# --- Layout ---
layout = html.Div(style=NEUMORPHISM_STYLE, children=[
    html.H1("A2 Model Predictor", style={"textAlign": "center", "marginBottom": "24px"}),

    # Modal popup
    html.Div(id="popup-modal", style=MODAL_STYLE, children=[
        html.Div(style=MODAL_CONTENT_STYLE, children=[
            html.H2("Welcome to the New Model!", style={"marginBottom": "12px"}),
            html.P(
        "This model has been developed after extensive experimentation and provides more accurate predictions than the old model.",
        style={"marginBottom": "12px", "textAlign": "left"}
    ),
    html.P("Key Features of the New Model:", style={"fontWeight": "600", "marginBottom": "8px", "textAlign": "left"}),
    html.Ul([
        html.Li("Feature Transformation: Polynomial"),
        html.Li("Degree of Polynomial: 2"),
        html.Li("Initialization: Zero Initialization"),
        html.Li("Optimization: Mini Batch Gradient Descent"),
        html.Li("Momentum: With Momentum"),
        html.Li("Learning Rate: 0.01")
    ], style={"paddingLeft": "20px", "textAlign": "left"}),
            html.Button("Got it!", id="close-modal", style=BUTTON_STYLE)
        ])
    ]),

    html.Div(style=CARD_STYLE, children=[
        html.Div(style=ROW_STYLE, children=[
            html.Div(children=[html.Label("Year of Manufacture", style=LABEL_STYLE),
                               dcc.Input(id="new-year", type="number", value=2019, step=1, style=INPUT_STYLE)]),
            html.Div(children=[html.Label("Max Power (bhp)", style=LABEL_STYLE),
                                dcc.Input(id="new-max_power", type="text", value=94.5, step=1, style=INPUT_STYLE)]),
            html.Div(children=[html.Label("Mileage (kmpl)", style=LABEL_STYLE),
                                dcc.Input(id="new-mileage", type="text", value=14.6, step=0.1, style=INPUT_STYLE)]),
        ]),
        html.Button("Predict Price", id="new-predict-button", n_clicks=0, style=BUTTON_STYLE),
        html.Div(id="new-prediction-output", style=OUTPUT_STYLE),
    ]),
])

# --- Callbacks ---
@dash.callback(
    Output("new-prediction-output", "children"),
    Input("new-predict-button", "n_clicks"),
    State("new-year", "value"),
    State("new-max_power", "value"),
    State("new-mileage", "value"),
)
def predict_price_new(n_clicks, year, max_power, mileage):
    if n_clicks <= 0:
        return "Enter details and click Predict."
    year = int(year)
    max_power = float(max_power)
    mileage = float(mileage)
    X_input = pd.DataFrame([[year, max_power, mileage]], columns=['year', 'max_power', 'mileage'])
    X_scaled = scaler.transform(X_input)
    log_pred = model.predict(X_scaled)[0]
    price = float(np.exp(log_pred))
    return f"Estimated Selling Price: ₹{price:,.0f}"

# Callback to close modal
@dash.callback(
    Output("popup-modal", "style"),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True
)
def close_modal(n_clicks):
    return {"display": "none"}