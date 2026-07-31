
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Customer Transaction Prediction",
    page_icon="💳",
    layout="centered"
)

# =========================================================
# PROJECT PATH
# =========================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "customer_transaction_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "train.csv"
)

# =========================================================
# TITLE
# =========================================================
st.title("💳 Customer Transaction Prediction")

st.write(
    "Enter the customer's ID_Code to predict "
    "whether the customer will make a transaction."
)

st.info(
    "Example ID_Code: train_0, train_1, train_2"
)

# =========================================================
# LOAD MODEL
# =========================================================
try:

    model = joblib.load(
        MODEL_PATH
    )

except Exception as e:

    st.error(
        "❌ Could not load the model."
    )

    st.write(
        "Make sure customer_transaction_model.pkl "
        "is in the same folder as app.py."
    )

    st.exception(e)

    st.stop()

# =========================================================
# LOAD SCALER
# =========================================================
try:

    scaler = joblib.load(
        SCALER_PATH
    )

except Exception as e:

    st.error(
        "❌ Could not load the scaler."
    )

    st.write(
        "Make sure scaler.pkl "
        "is in the same folder as app.py."
    )

    st.exception(e)

    st.stop()

# =========================================================
# LOAD DATASET
# =========================================================
try:

    train = pd.read_csv(
        DATA_PATH
    )

except Exception as e:

    st.error(
        "❌ Could not load train.csv."
    )

    st.write(
        "Make sure train.csv is inside the Data folder."
    )

    st.code(
        DATA_PATH
    )

    st.exception(e)

    st.stop()

# =========================================================
# ID_CODE INPUT
# =========================================================
id_code = st.text_input(
    "Enter ID_Code",
    placeholder="Example: train_0"
)

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button(
    "🔍 Predict Transaction",
    use_container_width=True
):

    # -----------------------------------------------------
    # CHECK EMPTY INPUT
    # -----------------------------------------------------
    if id_code.strip() == "":

        st.warning(
            "⚠️ Please enter an ID_Code."
        )

    else:

        # Remove spaces
        id_code = id_code.strip()

        # -------------------------------------------------
        # FIND CUSTOMER
        # -------------------------------------------------
        customer = train[
            train["ID_code"] == id_code
        ]

        # -------------------------------------------------
        # CHECK ID
        # -------------------------------------------------
        if customer.empty:

            st.error(
                f"❌ ID_Code '{id_code}' was not found."
            )

            st.write(
                "Example available ID_Codes:"
            )

            st.write(
                train["ID_code"].head(10).tolist()
            )

        else:

            st.success(
                f"✅ Customer '{id_code}' found."
            )

            # -------------------------------------------------
            # GET 200 FEATURES
            # -------------------------------------------------
            X_customer = customer.drop(
                columns=[
                    "ID_code",
                    "target"
                ],
                errors="ignore"
            )

            # -------------------------------------------------
            # SCALE FEATURES
            # -------------------------------------------------
            X_customer_scaled = scaler.transform(
                X_customer
            )

            # -------------------------------------------------
            # PREDICT
            # -------------------------------------------------
            prediction = model.predict(
                X_customer_scaled
            )[0]

            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------
            st.subheader(
                "Prediction Result"
            )

            st.write(
                f"**ID_Code:** {id_code}"
            )

            if prediction == 1:

                st.success(
                    "✅ Transaction"
                )

                st.write(
                    "The customer is predicted "
                    "to make a transaction."
                )

            else:

                st.error(
                    "❌ No Transaction"
                )

                st.write(
                    "The customer is predicted "
                    "not to make a transaction."
                )

