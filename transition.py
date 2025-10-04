def register_model_to_production():
    import os
    import warnings
    import urllib3
    from mlflow import MlflowClient

    warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

    # Set MLflow authentication
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
    os.environ['MLFLOW_TRACKING_URI'] = "https://mlflow.ml.brain.cs.ait.ac.th"

    model_name = os.getenv("APP_MODEL_NAME", "st125998-a3-model")

    client = MlflowClient()
    print(f"Looking for model: {model_name}")

    try:
        registered_model = client.get_registered_model(model_name)
    except Exception as e:
        raise RuntimeError(f"Model '{model_name}' not found in MLflow registry: {e}")

    for mv in registered_model.latest_versions:  # type: ignore
        if mv.current_stage == "Staging":
            version = mv.version
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"Model version {version} promoted to Production")
            break
    else:
        print("No model found in Staging stage to promote.")
