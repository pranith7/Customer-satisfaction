import numpy as np
import pandas as pd
from materializer.custom_materializer import cs_materializer 

from zenml import step,pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.intergrations.constants import mlflow 
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLflowModelDeployer
from zenml.integrations.mlflow.steps.mlflow.services import MLflowModelDeployerService
from zenml.integrations.mlflow.steps.mlflow_steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters,Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.train_model import train_model

docker_settings = DockerSettings(required_integrations=[mlflow])







