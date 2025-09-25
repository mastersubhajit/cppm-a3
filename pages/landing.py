import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Div([
    # Hero Section
    html.Div([
        html.Div([
            html.H1("Car Price Predictor", className="hero-title"),
            html.P("AI-powered predictions for your dream cars.", className="hero-subtitle"),
            # Link updated to A3 Model
            dcc.Link("Get Started", href="/a3-model", className="hero-button"),
        ], className="hero-content")
    ], className="hero-section"),

    # Features Section
    html.Div([
        html.H2("Why Choose Us?", className="section-title"),
        html.Div([
            html.Div([
                html.Img(src="/assets/Luxury.webp", className="feature-img"),
                html.H4("Luxury Cars"),
                html.P("Explore predictions for premium supercars.")
            ], className="feature-card"),

            html.Div([
                html.Img(src="/assets/Audi.webp", className="feature-img"),
                html.H4("Advanced AI Models"),
                html.P("Compare predictions from multiple trained models.")
            ], className="feature-card"),

            html.Div([
                html.Img(src="/assets/Sketch.webp", className="feature-img"),
                html.H4("Data Driven"),
                html.P("Powered by real car market data.")
            ], className="feature-card"),
        ], className="features-grid"),
    ], className="features-section"),

    # Footer
    html.Footer([
        html.P("© 2025 Car Price Predictor | Built with Dash + Neumorphism UI")
    ], className="footer")
])