# pages/a3_model.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import mlflow
import warnings
import cloudpickle
import os

# ===== Register page with Dash Pages =====
dash.register_page(__name__, path="/a3-model", name="A3 Model", title="A3 Model Predictor (Classification)")

# --- MLflow Config ---
mlflow.set_tracking_uri("https://admin:password@mlflow.ml.brain.cs.ait.ac.th")
model_name = "st125998-a3-model"
model_version = 24

# --- FIXED FEATURE AND BRAND LISTS ---
CORE_FEATURES = ['year', 'engine', 'max_power', 'mileage']
# IMPORTANT: use the same order as in training
NUMERIC_COLS_ORDER = ['year', 'max_power', 'mileage', 'engine']

BRAND_LIST = [
    'Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 
    'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 
    'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 
    'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'
]
ALL_FEATURES = CORE_FEATURES + [f'brand_{b}' for b in BRAND_LIST]

# --- Load Model and Scaler ---
model = None
scaler = None

try:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    print(f"Successfully loaded MLflow model: {model_name} v{model_version}")

    # Build absolute path to scaler file
    current_dir = os.path.dirname(os.path.abspath(__file__))          # /app/pages
    project_root = os.path.abspath(os.path.join(current_dir, ".."))   # /app
    scaler_path = os.path.join(project_root, "models", "cppm_a3_scaler.pkl")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = cloudpickle.load(f)
    print(f"Successfully loaded StandardScaler from {scaler_path}")

except Exception as e:
    print(f"ERROR loading model or scaler: {e}")
    class DummyModel:
        def predict(self, X): return np.array([3])
    model = DummyModel()
    scaler = None

# --- Dummy DataFrame for Default Values ---
df = pd.DataFrame({
    'year': [2019], 
    'engine': [1197.0], 
    'max_power': [94.5], 
    'mileage': [14.6]
})

# --- Defaults ---
DEFAULT_YEAR = int(df['year'].mode()[0]) if not df.empty else 2019
DEFAULT_ENGINE = float(df['engine'].median()) if not df.empty else 1197.0
DEFAULT_POWER = float(df['max_power'].median()) if not df.empty else 94.5
DEFAULT_MILEAGE = float(df['mileage'].median()) if not df.empty else 14.6
DEFAULT_BRAND = BRAND_LIST[0] if BRAND_LIST else 'Maruti'

# --- Styles ---
NEUMORPHISM_STYLE = {"background": "#fff", "minHeight": "100vh", "padding": "40px", "fontFamily": "Inter, sans-serif", "color": "#2c3e50"}
CARD_STYLE = {"background": "#e0e5ec", "boxShadow": "9px 9px 18px #a3b1c6, -9px -9px 16px #ffffff", "borderRadius": "20px", "padding": "28px", "margin": "0 auto", "maxWidth": "860px"}
ROW_FULL_STYLE = {"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "16px", "marginTop": "12px", "alignItems": "start"}
LABEL_STYLE = {"fontSize": "14px", "marginBottom": "6px", "display": "block", "textAlign": "center"}
INPUT_STYLE = {
    "textAlign": "center",
    "background": "#e0e5ec",
    "boxShadow": "inset 5px 5px 10px #a3b1c6, inset -5px -5px 10px #ffffff",
    "border": "none",
    "borderRadius": "12px",
    "padding": "12px",
    "width": "100%",  
    "maxWidth": "200px", 
    "outline": "none",
    "fontSize": "14px",
    "display": "block",
    "margin": "0 auto" 
}
DROPDOWN_STYLE = {
    "background": "#e0e5ec",
    "boxShadow": "inset 5px 5px 10px #a3b1c6, inset -5px -5px 10px #ffffff",
    "border": "none",
    "borderRadius": "12px",
    "padding": "0px",
    "width": "100%", 
    "maxWidth": "200px", 
    "outline": "none",
    "fontSize": "14px",
    "margin": "0 auto" 
}
BUTTON_STYLE = {"background": "#e0e5ec", "boxShadow": "5px 5px 10px #a3b1c6, -5px -5px 10px #ffffff", "border": "none", "borderRadius": "12px", "padding": "14px 24px", "cursor": "pointer", "fontWeight": "600", "width": "220px", "marginTop": "28px", "display": "block", "margin": "20px auto", "textAlign": "center"}
OUTPUT_STYLE = {"background": "#e0e5ec", "boxShadow": "inset 6px 6px 12px #a3b1c6, inset -6px -6px 12px #ffffff", "borderRadius": "15px", "padding": "18px", "marginTop": "20px", "textAlign": "center", "fontSize": "20px", "fontWeight": "700"}

# --- Layout ---
layout = html.Div(style=NEUMORPHISM_STYLE, children=[
    html.H1("A3 Model Predictor (Classification)", style={"textAlign": "center", "marginBottom": "24px"}),

    html.Div(style=CARD_STYLE, children=[
        # Row 1: Brand (Centered and spanning 2 columns)
        html.Div(style=ROW_FULL_STYLE, children=[
            html.Div(),
            html.Div(style={"gridColumn": "span 2"}, children=[ 
                html.Label("Car Brand", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="a3-brand",
                    options=[{'label': brand, 'value': brand} for brand in BRAND_LIST],
                    value=DEFAULT_BRAND,
                    clearable=False,
                    style=DROPDOWN_STYLE,
                    className="dropdown-neumorphism" 
                )
            ]),
            html.Div(), 
        ]),

        # Row 2: Numerical Features
        html.Div(style=ROW_FULL_STYLE, children=[
            html.Div(children=[html.Label("Year", style=LABEL_STYLE),
                               dcc.Input(id="a3-year", type="number", value=DEFAULT_YEAR, step=1, style=INPUT_STYLE)]),
            html.Div(children=[html.Label("Engine (CC)", style=LABEL_STYLE),
                               dcc.Input(id="a3-engine", type="number", value=DEFAULT_ENGINE, step=1, style=INPUT_STYLE)]),
            html.Div(children=[html.Label("Max Power (bhp)", style=LABEL_STYLE),
                               dcc.Input(id="a3-max_power", type="number", value=DEFAULT_POWER, step=0.1, style=INPUT_STYLE)]),
            html.Div(children=[html.Label("Mileage (kmpl)", style=LABEL_STYLE),
                               dcc.Input(id="a3-mileage", type="number", value=DEFAULT_MILEAGE, step=0.1, style=INPUT_STYLE)]),
        ]),

        html.Button("Predict Class", id="a3-predict-button", n_clicks=0, style=BUTTON_STYLE),
        html.Div(id="a3-prediction-output", style=OUTPUT_STYLE),
    ]),
])

# --- Callback ---
@dash.callback(
    Output("a3-prediction-output", "children"),
    Input("a3-predict-button", "n_clicks"),
    State("a3-year", "value"),
    State("a3-engine", "value"),
    State("a3-max_power", "value"),
    State("a3-mileage", "value"),
    State("a3-brand", "value"),
)
def predict_price_a3(n_clicks, year, engine, max_power, mileage, brand):
    if n_clicks <= 0:
        return "Enter details and click Predict."
    
    if None in [year, engine, max_power, mileage, brand]:
        return "All fields are required. Please fill in every input."

    try:
        # Convert to numeric
        year = int(year)
        engine = float(engine)
        max_power = float(max_power)
        mileage = float(mileage)

        # Build input dict
        X_input_dict = {feat: 0 for feat in ALL_FEATURES}
        X_input_dict['year'] = year
        X_input_dict['engine'] = engine
        X_input_dict['max_power'] = max_power
        X_input_dict['mileage'] = mileage

        # One-hot encode brand
        ohe_col = f'brand_{brand}'
        if ohe_col in ALL_FEATURES:
            X_input_dict[ohe_col] = 1

        # Create DataFrame
        X_df = pd.DataFrame([X_input_dict], columns=ALL_FEATURES)

        # Scale only numeric features (respect training order)
        if scaler is not None:
            X_df[NUMERIC_COLS_ORDER] = scaler.transform(X_df[NUMERIC_COLS_ORDER])
        else:
            return "Error: Scaler not loaded. Cannot scale input."

        # Convert to numpy (float64) before prediction
        X_array = X_df.to_numpy().astype(np.float64)

        # Predict
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            predicted_class = model.predict(X_array)

        predicted_class_str = str(predicted_class[0])
        return html.Div([
            html.P("Predicted Selling Price Class:", 
                   style={'fontSize': '16px', 'fontWeight': '400', 'marginBottom': '5px'}),
            html.H3(predicted_class_str, style={'margin': '0', 'color': '#2c3e50'})
        ])
    
    except (ValueError, TypeError):
        return "Invalid input. Please ensure all numerical fields are valid numbers."
    except Exception as e:
        return f"Prediction Error: {str(e)}"