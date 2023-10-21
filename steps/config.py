from zenml.steps import BaseParameters
# from zenml.step.base_step import BaseStep



class ModelNameConfig(BaseParameters):
    """model config"""
    model_name: str = "LinearRegression"
