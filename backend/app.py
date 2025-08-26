import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import euclidean_distances

# ------------------------------------------------------
# Load trained model and scaler (export these in notebook):
#   joblib.dump(rfr, "model.pkl")
#   joblib.dump(scaler, "scaler.pkl")
# ------------------------------------------------------
model = joblib.load("/Users/mastersubhajitghosh/Downloads/Machine_Learning/Predicting_Car_Prices/backend/model/car_price_prediction_model.pkl")
scaler = joblib.load("/Users/mastersubhajitghosh/Downloads/Machine_Learning/Predicting_Car_Prices/backend/model/scaler.pkl")

# ------------------------------------------------------
# Load and preprocess dataset for similarity lookup
# ------------------------------------------------------
df = pd.read_csv("/Users/mastersubhajitghosh/Downloads/Machine_Learning/Predicting_Car_Prices/backend/data/Cars.csv")

# Keep human-readable owner for display; numeric map only if needed elsewhere
if "owner" in df.columns:
    df["owner_display"] = df["owner"]
    owner_map = {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
        "Test Drive Car": 5
    }
    # If your notebook used numeric owner, keep it mapped; display uses owner_display
    df["owner_num"] = df["owner"].map(owner_map)

# Clean mileage ("18.6 kmpl" -> 18.6)
if "mileage" in df.columns:
    df["mileage"] = df["mileage"].astype(str).str.split().str[0]
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

# Clean max_power ("74 bhp" -> 74.0)
if "max_power" in df.columns:
    df["max_power"] = df["max_power"].astype(str).str.split().str[0]
    df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")

# Extract brand from name (e.g., "Maruti Swift Dzire" -> "Maruti")
if "name" in df.columns:
    df["brand"] = df["name"].astype(str).str.split().str[0]

# Ensure required numeric features exist and are valid
features = ["year", "max_power", "mileage"]
for col in features:
    if col not in df.columns:
        raise ValueError(f"Required feature '{col}' not found in Cars.csv")

df = df.dropna(subset=features)

# Drop duplicates based on representative identity
dedupe_keys = [c for c in ["name", "brand", "year", "km_driven", "fuel", "transmission", "owner_display"] if c in df.columns]
if dedupe_keys:
    df = df.drop_duplicates(subset=dedupe_keys).reset_index(drop=True)

# ------------------------------------------------------
# Dash app with Neumorphism UI
# ------------------------------------------------------
app = dash.Dash(__name__)
server = app.server

# Neumorphism styles
NEUMORPHISM_STYLE = {
    "background": "#e0e5ec",
    "minHeight": "100vh",
    "padding": "40px",
    "fontFamily": "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
    "color": "#2c3e50",
}

CARD_STYLE = {
    "background": "#e0e5ec",
    "boxShadow": "9px 9px 16px #a3b1c6, -9px -9px 16px #ffffff",
    "borderRadius": "20px",
    "padding": "28px",
    "margin": "0 auto",
    "maxWidth": "760px",
}

ROW_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr 1fr",
    "gap": "16px",
    "marginTop": "12px",
    "alignItems": "center",
}

LABEL_STYLE = {
    "fontSize": "14px",
    "marginBottom": "6px",
    "display": "block",
}

INPUT_STYLE = {
    "textAlign": "center",
    "background": "#e0e5ec",
    "boxShadow": "inset 5px 5px 10px #a3b1c6, inset -5px -5px 10px #ffffff",
    "border": "none",
    "borderRadius": "12px",
    "padding": "12px",
    "width": "200px",
    "outline": "none",
    "fontSize": "14px",
}

BUTTON_STYLE = {
    "background": "#e0e5ec",
    "boxShadow": "5px 5px 10px #a3b1c6, -5px -5px 10px #ffffff",
    "border": "none",
    "borderRadius": "12px",
    "padding": "14px 24px",
    "cursor": "pointer",
    "fontWeight": "600",
    "width": "200px",
    "marginTop": "18px",
    "display": "block",          
    "margin": "20px auto",        
    "textAlign": "center",
}

OUTPUT_STYLE = {
    "background": "#e0e5ec",
    "boxShadow": "inset 6px 6px 12px #a3b1c6, inset -6px -6px 12px #ffffff",
    "borderRadius": "15px",
    "padding": "18px",
    "marginTop": "20px",
    "textAlign": "center",
    "fontSize": "20px",
    "fontWeight": "700",
}

TABLE_WRAP_STYLE = {
    "background": "#e0e5ec",
    "boxShadow": "inset 6px 6px 12px #a3b1c6, inset -6px -6px 12px #ffffff",
    "borderRadius": "16px",
    "padding": "16px",
    "marginTop": "28px",
}

TABLE_STYLE = {
    "width": "100%",
    "borderCollapse": "separate",
    "borderSpacing": "0 10px",
    "fontSize": "14px",
}

TH_STYLE = {
    "textAlign": "left",
    "padding": "12px 14px",
    "opacity": "0.85",
}

TD_STYLE = {
    "background": "#e0e5ec",
    "boxShadow": "5px 5px 10px #a3b1c6, -5px -5px 10px #ffffff",
    "borderRadius": "12px",
    "padding": "12px 14px",
}

# Layout
app.layout = html.Div(style=NEUMORPHISM_STYLE, children=[
    html.H1("Car Price Predictor", style={"textAlign": "center", "marginBottom": "24px"}),

    html.Div(style=CARD_STYLE, children=[
        html.Div(style=ROW_STYLE, children=[
            html.Div(children=[
                html.Label("Year of Manufacture", style=LABEL_STYLE),
                dcc.Input(id="year", type="number", value=2019, step=1, style=INPUT_STYLE),
            ]),
            html.Div(children=[
                html.Label("Max Power (bhp)", style=LABEL_STYLE),
                dcc.Input(id="max_power", type="number", value=90, step=1, style=INPUT_STYLE),
            ]),
            html.Div(children=[
                html.Label("Mileage (kmpl)", style=LABEL_STYLE),
                dcc.Input(id="mileage", type="number", value=18.0, step=0.1, style=INPUT_STYLE),
            ]),
        ]),

        html.Button("Predict Price", id="predict-button", n_clicks=0, style=BUTTON_STYLE),

        html.Div(id="prediction-output", style=OUTPUT_STYLE),

        html.Div(id="similar-cars-output", style=TABLE_WRAP_STYLE),
    ]),
])

# Callback
@app.callback(
    [Output("prediction-output", "children"),
     Output("similar-cars-output", "children")],
    Input("predict-button", "n_clicks"),
    State("year", "value"),
    State("max_power", "value"),
    State("mileage", "value"),
)
def predict_price(n_clicks, year, max_power, mileage):
    if n_clicks <= 0:
        return "Enter details and click Predict.", ""

    # Validate inputs
    if year is None or max_power is None or mileage is None:
        return "Please provide all inputs.", ""

    # Prediction
    X_input = np.array([[year, max_power, mileage]])
    X_scaled = scaler.transform(X_input)
    log_pred = model.predict(X_scaled)[0]
    price = float(np.exp(log_pred))

    # Similarity lookup
    df_filtered = df.dropna(subset=features).copy()
    dataset_scaled = scaler.transform(df_filtered[features])
    dists = euclidean_distances(X_scaled, dataset_scaled)[0]
    df_filtered["distance"] = dists

    show_cols = []
    for c in ["brand", "name", "year", "km_driven", "fuel", "transmission", "owner_display", "selling_price"]:
        if c in df_filtered.columns:
            show_cols.append(c)

    similar = df_filtered.nsmallest(5, "distance")[show_cols].copy()

    # Rename owner_display -> owner for human-readable output
    if "owner_display" in similar.columns:
        similar = similar.rename(columns={"owner_display": "owner"})

    # Build Neumorphic table
    header = html.Tr([html.Th(col, style=TH_STYLE) for col in similar.columns])
    body_rows = []
    for _, row in similar.iterrows():
        tds = [html.Td(row[col], style=TD_STYLE) for col in similar.columns]
        body_rows.append(html.Tr(tds, style={"height": "48px"}))

    table = html.Table(
        children=[html.Thead(header), html.Tbody(body_rows)],
        style=TABLE_STYLE,
    )

    price_text = f"Estimated Selling Price: ₹{price:,.0f}"
    return price_text, html.Div([
        html.H3("Similar Cars that Match Your Criteria", style={"margin": "6px 4px 14px 4px", "textAlign": "center"}),
        table
    ])

if __name__ == "__main__":
    app.run(debug=True, port=8080)
