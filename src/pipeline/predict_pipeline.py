import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

num = [
    "term",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type",
]

cat = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
]


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

import pandas as pd
import sys


class CustomData:
    def __init__(
        self,
        term: int,
        grade: str,
        home_ownership: str,
        verification_status: str,
        purpose: str,
        initial_list_status: str,
        application_type: str,
        loan_amnt: float,
        int_rate: float,
        installment: float,
        annual_inc: float,
        dti: float,
        open_acc: int,
        pub_rec: int,
        revol_bal: float,
        revol_util: float,
        total_acc: int,
        mort_acc: int,
        pub_rec_bankruptcies: int,
    ):
        self.term = term
        self.grade = grade
        self.home_ownership = home_ownership
        self.verification_status = verification_status
        self.purpose = purpose
        self.initial_list_status = initial_list_status
        self.application_type = application_type
        self.loan_amnt = loan_amnt
        self.int_rate = int_rate
        self.installment = installment
        self.annual_inc = annual_inc
        self.dti = dti
        self.open_acc = open_acc
        self.pub_rec = pub_rec
        self.revol_bal = revol_bal
        self.revol_util = revol_util
        self.total_acc = total_acc
        self.mort_acc = mort_acc
        self.pub_rec_bankruptcies = pub_rec_bankruptcies

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "term": [self.term],
                "grade": [self.grade],
                "home_ownership": [self.home_ownership],
                "verification_status": [self.verification_status],
                "purpose": [self.purpose],
                "initial_list_status": [self.initial_list_status],
                "application_type": [self.application_type],
                "loan_amnt": [self.loan_amnt],
                "int_rate": [self.int_rate],
                "installment": [self.installment],
                "annual_inc": [self.annual_inc],
                "dti": [self.dti],
                "open_acc": [self.open_acc],
                "pub_rec": [self.pub_rec],
                "revol_bal": [self.revol_bal],
                "revol_util": [self.revol_util],
                "total_acc": [self.total_acc],
                "mort_acc": [self.mort_acc],
                "pub_rec_bankruptcies": [self.pub_rec_bankruptcies],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
