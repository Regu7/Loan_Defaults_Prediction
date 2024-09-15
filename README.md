# End to End Machine Learning Project - Loan Default Prediction for LendingClub

## Problem Statement

**Loan Default Prediction for LendingClub**

LendingClub is a leading peer-to-peer lending company that connects borrowers with investors. The company needs to make informed decisions about loan approvals based on the applicant’s profile to minimize financial risks. The primary objective of this project is to predict whether a loan applicant will fully repay the loan or default (charged-off).

### Business Context

When a loan application is received, LendingClub faces two types of risks:

- **Loss of Business**: If an applicant who is likely to repay the loan is not approved, the company loses potential business.
- **Financial Loss**: If an applicant who is likely to default is approved, the company incurs a financial loss.

### Objective

The goal is to develop a predictive model using historical data of past loan applicants to identify patterns that indicate whether a person is likely to default. This model will help LendingClub make data-driven decisions to:

- Approve or deny loans
- Adjust loan amounts
- Set appropriate interest rates for risky applicants



## Steps to Run the App using this repo:

From the project root folder

1. Create a conda and environment and install the requirement file

`pip install -r deploy\requirements.text`

2. Train the model

`python src\pipeline\train_pipeline.py`

3. Run the streamlit app

`streamlit run app.py`

## Steps to run the app using the docker Image

1. Download the docker image

`docker pull reguh/loan-default-prediction:latest`

2. Create a container

`docker run -p 8501:8501 --name loan_def loan-default-prediction:latest`