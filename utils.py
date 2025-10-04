import cloudpickle

def save(filename:str, obj:object):
    with open(filename, 'wb') as handle:
        cloudpickle.dump(obj, handle, protocol=cloudpickle.HIGHEST_PROTOCOL)

def load(filename:str) -> object:
    with open(filename, 'rb') as handle:
        b = cloudpickle.load(handle)
    return b

import mlflow
import os

def load_mlflow(stage='Staging'):
    model_name = os.environ['APP_MODEL_NAME']
    cache_path = os.path.join("models",stage)
    if(os.path.exists(cache_path) == False):
        os.makedirs(cache_path)
    
    # check if we cache the model
    path = os.path.join(cache_path,model_name)
    if(os.path.exists( path ) == False):
        # This will keep load the model again and again.
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
        save(filename=path, obj=model)

    model = load(path)
    return model

def register_model_to_production():
    from mlflow.client import MlflowClient
    from mlflow.exceptions import RestException
    import os
    
    # Set MLflow authentication
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
    
    model_name = os.environ['APP_MODEL_NAME']
    client = MlflowClient()
    
    try:
        registered_model = client.get_registered_model(model_name)
        staging_found = False
        
        for model in registered_model.latest_versions:
            if model.current_stage == "Staging":
                staging_found = True
                version = model.version
                client.transition_model_version_stage(
                    name=model_name, version=version, stage="Production", archive_existing_versions=True
                )
                print(f"Model version {version} is promoted to Production")
                break
        
        if not staging_found:
            print("No model found in Staging stage")
            
    except RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            print(f"Model not found in registry")
        else:
            raise e