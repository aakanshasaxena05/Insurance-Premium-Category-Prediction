import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.title("Insurance Premium Category Predictor")
st.write("Enter your details below:")


# -----------------------------
# INPUT FIELDS
# -----------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=119,
    value=30
)

weight = st.number_input(
    "Weight (kg)",
    min_value=1.0,
    max_value=300.0,
    value=65.0
)

height = st.number_input(
    "Height (cm)",
    min_value=50.0,
    max_value=250.0,
    value=170.0
)

income_lpa = st.number_input(
    "Annual Income (LPA)",
    min_value=0.1,
    max_value=1000.0,
    value=10.0
)

smoker = st.selectbox(
    "Are you a smoker?",
    ["Yes", "No"]
)

city = st.text_input(
    "City",
    value="Mumbai"
)

occupation = st.selectbox(
    "Occupation",
    [
        "Student",
        "Software Engineer",
        "Business",
        "Teacher"
    ]
)


# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Premium Category"):

    input_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    st.write("### Data sent to FastAPI")
    st.json(input_data)

    try:

        response = requests.post(
            API_URL,
            json=input_data,
            timeout=20
        )

        st.write("API Status Code:", response.status_code)

        # Successful response
        if response.status_code == 200:

            result = response.json()

            st.write("### API Response")
            st.json(result)

            prediction = result["predicted_category"]

            st.success(
                f"Predicted Insurance Premium Category: **{prediction}**"
            )

        # API returned an error
        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.write(response.text)

    # FastAPI cannot be reached
    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the FastAPI server."
        )

        st.info(
            "Check whether http://34.226.152.222:8000 is reachable."
        )

    except requests.exceptions.Timeout:

        st.error(
            "❌ FastAPI server took too long to respond."
        )

    except Exception as e:

        st.error(
            f"❌ Error: {e}"
        )