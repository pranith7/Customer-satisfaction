import logging

import pandas as pd
from zenml import step 
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.DataFrame,"y_train"],
    Annotated[pd.DataFrame,"y_test"],
    ]:
    """
    Cleans the data and divides it into train ans test

    Args:
        df: Raw data
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train:Training labels
        y_test: Testing labels 
    """

    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df,process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data,divide_strategy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        y_train = y_train.to_frame(name="y_train")  # Convert Series to DataFrame
        y_test = y_test.to_frame(name="y_test")  # Convert Series to DataFrame
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e


