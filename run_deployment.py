from pipelines.deployment_pipeline import deploy_pipeline,inference_pipeline
import click

DEPLOY = "deploy"
DEPLOY_AND_PREDICT = "deploy_and_predict"
PREDICT = "predict"



@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model (`deploy`), or to "
    "only run a prediction against the deployed model "
    "(`predict`). By default both will be run "
    "(`deploy_and_predict`).",
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model",
)

def run_pipeline(config: str, min_accuracy: float):
    if config == DEPLOY:
        deploy_pipeline(min_accuracy=min_accuracy)
    elif config == PREDICT:
        inference_pipeline()
        
        # deploy_pipeline(min_accuracy=min_accuracy)
    # elif config == DEPLOY_AND_PREDICT:
        # deploy_pipeline(min_accuracy=min_accuracy)
        # inference_pipeline()
