import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info(f"Current working dir : {os.getcwd()}")
            logging.info(f"Read the dataset as dataframe")
            df = pd.read_csv(
                os.path.join(
                    # os.path.dirname(os.path.abspath(__file__)),
                    r"data\accepeted_custom.csv",
                )
            )
            # df = pd.read_csv(r"data\accepeted_custom.csv")
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]

            # percentage = 1

            # # Sample the DataFrame
            # df = df.sample(frac=percentage, random_state=42)

            logging.info("Data sampled and took only {percentage} rows")

            # cols to drop based on EDA and Model analysis
            drop_cols = [
                "earliest_cr_line",
                "issue_d",
                "zip_code",
                "addr_state",
                "emp_title",
                "emp_length",
                "title",
                "sub_grade",
            ]

            logging.info("Dropped Unecessary Columns")
            df.drop(columns=drop_cols, inplace=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Data Ingestion is completed")

            print(self.ingestion_config.test_data_path)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
