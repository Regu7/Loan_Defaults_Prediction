import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_trans_cfg = DataTransformationConfig

    def get_data_transformer_object(self, numerical_cols, cat_cols):
        """
        This function is resposnible for data transfoirmation

        """
        try:

            # def map_term(df):
            #     df['term'] = df['term'].map({" 36 months": 36, " 60 months": 60})
            #     return df

            # def transform_home_ownership(df):
            #     df.loc[(df.home_ownership == "ANY") | (df.home_ownership == "NONE"), "home_ownership"] = "OTHER"
            #     return df

            # term_transformer = FunctionTransformer(map_term)
            # home_ownership_transformer = FunctionTransformer(transform_home_ownership)

            num_cols_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("standarscaler", StandardScaler(with_mean=False)),
                ]
            )

            cat_cols_pipe = Pipeline(
                steps=[
                    # ("term_transformer", term_transformer),
                    # ("home_ownership_transformer", home_ownership_transformer),
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder()),
                    ("standard_scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Numerical columns : {numerical_cols}")
            logging.info(f"Categorical columns : {cat_cols}")

            preprocessor = ColumnTransformer(
                [
                    ("num_cols_pipe1", num_cols_pipe, numerical_cols),
                    ("cat_cols_pipe1", cat_cols_pipe, cat_cols),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            target_column_name = "loan_status"

            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)
            logging.info("Drop the Nan values")

            if pd.api.types.is_numeric_dtype(train_df[target_column_name]):
                pass
            else:
                train_df[target_column_name] = train_df[target_column_name].map(
                    {"Fully Paid": 0, "Charged Off": 1}
                )
            if pd.api.types.is_numeric_dtype(test_df[target_column_name]):
                pass
            else:
                test_df[target_column_name] = test_df[target_column_name].map(
                    {"Fully Paid": 0, "Charged Off": 1}
                )

            train_df.term = train_df.term.map({" 36 months": 36, " 60 months": 60})
            test_df.term = test_df.term.map({" 36 months": 36, " 60 months": 60})

            train_df.loc[
                (train_df.home_ownership == "ANY")
                | (train_df.home_ownership == "NONE"),
                "home_ownership",
            ] = "OTHER"

            test_df.loc[
                (test_df.home_ownership == "ANY") | (test_df.home_ownership == "NONE"),
                "home_ownership",
            ] = "OTHER"

            numerical_cols = train_df.select_dtypes(
                include=["int", "float"]
            ).columns.to_list()
            numerical_cols.remove(target_column_name)
            cat_cols = train_df.select_dtypes(
                include=["object", "category"]
            ).columns.to_list()

            preprocessing_obj = self.get_data_transformer_object(
                numerical_cols, cat_cols
            )

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_trans_cfg.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_trans_cfg.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
