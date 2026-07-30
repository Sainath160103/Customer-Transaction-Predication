
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Customer Transaction Prediction",
    page_icon="💳",
    layout="centered"
)

# =========================================================
# APPLICATION TITLE
# =========================================================
st.title("💳 Customer Transaction Prediction")

st.write(
    "Enter a binary value to check the transaction status."
)



# =========================================================
# USER INPUT
# =========================================================
user_input = st.selectbox(
    "Enter Customer Transaction Input",
    options=[0, 1]
)

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button("🔍 Predict Transaction", use_container_width=True):

    # -----------------------------------------------------
    # IF INPUT IS 1
    # -----------------------------------------------------
    if user_input == 1:

        st.success("✅ Transaction")

        st.write(
            "The customer transaction status is **Transaction**."
        )

    # -----------------------------------------------------
    # IF INPUT IS 0
    # -----------------------------------------------------
    else:

        st.error("❌ No Transaction")

        st.write(
            "The customer transaction status is **No Transaction**."
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "Customer Transaction Prediction System"
)