from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal
import pickle
import pandas as pd


# =========================================================
# LOAD MODEL
# =========================================================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# =========================================================
# CREATE FASTAPI APP
# =========================================================

app = FastAPI(
    title="Insurance Premium Prediction API",
    description="Predict insurance premium category using a trained Random Forest model",
    version="1.0"
)


# =========================================================
# CITY LISTS
# =========================================================

tier_1_cities = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Chennai",
    "Kolkata",
    "Hyderabad",
    "Pune"
]

tier_2_cities = [
    "Jaipur",
    "Chandigarh",
    "Indore",
    "Lucknow",
    "Patna",
    "Ranchi",
    "Visakhapatnam",
    "Coimbatore",
    "Bhopal",
    "Nagpur",
    "Vadodara",
    "Surat",
    "Rajkot",
    "Jodhpur",
    "Raipur",
    "Amritsar",
    "Varanasi",
    "Agra",
    "Dehradun",
    "Mysore",
    "Jabalpur",
    "Guwahati",
    "Thiruvananthapuram",
    "Ludhiana",
    "Nashik"
]


# =========================================================
# USER INPUT MODEL
# =========================================================

class UserInput(BaseModel):

    age: Annotated[
        int,
        Field(
            gt=0,
            lt=120,
            description="Age of the user"
        )
    ]

    weight: Annotated[
        float,
        Field(
            gt=0,
            description="Weight in kg"
        )
    ]

    height: Annotated[
        float,
        Field(
            gt=50,
            lt=250,
            description="Height in centimeters"
        )
    ]

    income_lpa: Annotated[
        float,
        Field(
            gt=0,
            description="Annual income in LPA"
        )
    ]

    smoker: Annotated[
        Literal["Yes", "No"],
        Field(
            description="Smoking status"
        )
    ]

    city: Annotated[
        str,
        Field(
            description="City of the user"
        )
    ]

    occupation: Annotated[
        Literal[
            "Student",
            "Software Engineer",
            "Business",
            "Teacher"
        ],
        Field(
            description="Occupation of the user"
        )
    ]


    # =====================================================
    # BMI
    # =====================================================

    @computed_field
    @property
    def bmi(self) -> float:

        # Convert cm to meters
        height_m = self.height / 100

        # BMI formula
        return self.weight / (height_m ** 2)


    # =====================================================
    # LIFESTYLE RISK
    # =====================================================

    @computed_field
    @property
    def lifestyle_risk(self) -> str:

        if self.smoker == "Yes" and self.bmi > 30:
            return "high"

        elif self.smoker == "Yes" and self.bmi > 27:
            return "medium"

        else:
            return "low"


    # =====================================================
    # AGE GROUP
    # =====================================================

    @computed_field
    @property
    def age_group(self) -> str:

        if self.age <= 18:
            return "Teen"

        elif self.age <= 45:
            return "Adult"

        else:
            return "Senior"


    # =====================================================
    # CITY TIER
    # =====================================================

    @computed_field
    @property
    def city_tier(self) -> str:

        if self.city in tier_1_cities:

            return "Tier 1"

        elif self.city in tier_2_cities:

            return "Tier 2"

        else:

            return "Tier 2"


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict_premium(data: UserInput):

    # Create dataframe containing exactly
    # the features used during model training

    input_df = pd.DataFrame([{

        "bmi": data.bmi,

        "age_group": data.age_group,

        "lifestyle_risk": data.lifestyle_risk,

        "city_tier": data.city_tier,

        "income_lpa": data.income_lpa,

        "occupation": data.occupation
    }])


    # Make prediction

    prediction = model.predict(input_df)[0]


    # Return result

    return JSONResponse(

        status_code=200,

        content={
            "predicted_category": str(prediction)
        }
    )