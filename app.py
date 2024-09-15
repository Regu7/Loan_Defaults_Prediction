import json
import logging

import pandas as pd
import streamlit as st

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# # Load the dataset
# df = pd.read_csv(r"artifacts\train.csv")

# # Extract unique values
# unique_values = {
#     "term": df["term"].unique().tolist(),
#     "grade": df["grade"].unique().tolist(),
#     "home_ownership": df["home_ownership"].unique().tolist(),
#     "verification_status": df["verification_status"].unique().tolist(),
#     "purpose": df["purpose"].unique().tolist(),
#     "initial_list_status": df["initial_list_status"].unique().tolist(),
#     "application_type": df["application_type"].unique().tolist(),
# }

# with open("artifacts/cat_unique_values.json", "w") as f:
#     json.dump(unique_values, f)


def main():
    st.title("Loan Default Prediction")

    with open("artifacts/cat_unique_values.json", "r") as f:
        unique_values = json.load(f)

    # Use the unique values in Streamlit
    term = st.selectbox("Term", unique_values["term"])
    grade = st.selectbox("Grade", unique_values["grade"])
    home_ownership = st.selectbox("Home Ownership", unique_values["home_ownership"])
    verification_status = st.selectbox(
        "Verification Status", unique_values["verification_status"]
    )
    purpose = st.selectbox("Purpose", unique_values["purpose"])
    initial_list_status = st.selectbox(
        "Initial List Status", unique_values["initial_list_status"]
    )
    application_type = st.selectbox(
        "Application Type", unique_values["application_type"]
    )

    loan_amnt = st.number_input("Loan Amount")
    int_rate = st.number_input("Interest Rate")
    installment = st.number_input("Installment")
    annual_inc = st.number_input("Annual Income")
    dti = st.number_input("Debt-to-Income Ratio")
    open_acc = st.number_input("Open Accounts")
    pub_rec = st.number_input("Public Records")
    revol_bal = st.number_input("Revolving Balance")
    revol_util = st.number_input("Revolving Utilization")
    total_acc = st.number_input("Total Accounts")
    mort_acc = st.number_input("Mortgage Accounts")
    pub_rec_bankruptcies = st.number_input("Public Record Bankruptcies")

    if st.button("Predict"):
        st.markdown("### Prediction Results")
        data = CustomData(
            term=term,
            grade=grade,
            home_ownership=home_ownership,
            verification_status=verification_status,
            purpose=purpose,
            initial_list_status=initial_list_status,
            application_type=application_type,
            loan_amnt=loan_amnt,
            int_rate=int_rate,
            installment=installment,
            annual_inc=annual_inc,
            dti=dti,
            open_acc=open_acc,
            pub_rec=pub_rec,
            revol_bal=revol_bal,
            revol_util=revol_util,
            total_acc=total_acc,
            mort_acc=mort_acc,
            pub_rec_bankruptcies=pub_rec_bankruptcies,
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        logging.info("Prediction input recieved and converted to a df")
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        logging.info("Initiating Prediction pipeline to get results")
        results = predict_pipeline.predict(pred_df)
        if results[0] == 0:
            results_str = "The loan applicant will fully pay the loan"
        else:
            results_str = "The loan applicant will likely default"

        logging.info(f"Predicted Output : {results}")
        print("after Prediction")
        st.write(f"{results_str}")


if __name__ == "__main__":
    main()
