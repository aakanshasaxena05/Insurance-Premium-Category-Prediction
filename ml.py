
# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score


# =========================================================
# 2. LOAD DATASET
# =========================================================

df = pd.read_csv(
    r"C:\Users\Lenovo\OneDrive\Desktop\fastapi1\insurance\insurance.csv",
    sep="|"
)


# =========================================================
# 3. CLEAN COLUMN NAMES
# =========================================================

# Remove spaces before and after column names
df.columns = df.columns.str.strip()


# Remove empty columns such as Unnamed: 0 and Unnamed: 9
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]


# =========================================================
# 4. CLEAN DATA VALUES
# =========================================================

# Remove extra spaces from string values
df = df.map(
    lambda x: x.strip() if isinstance(x, str) else x
)


# =========================================================
# 5. REMOVE MARKDOWN SEPARATOR ROW
# =========================================================

# The file contains a row such as:
# --:   -----:   -----:   ---------:
#
# We remove it by keeping only rows where age is numeric.

df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)

# Remove rows where age could not be converted
df = df.dropna(subset=["age"])


# Reset index
df = df.reset_index(drop=True)


# =========================================================
# 6. CONVERT NUMERIC COLUMNS
# =========================================================

df["age"] = pd.to_numeric(df["age"])
df["weight"] = pd.to_numeric(df["weight"])
df["height"] = pd.to_numeric(df["height"])
df["income_lpa"] = pd.to_numeric(df["income_lpa"])


# =========================================================
# 7. DISPLAY CLEAN DATA
# =========================================================

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nRandom 5 rows:")
print(df.sample(5))


# =========================================================
# 8. CREATE COPY
# =========================================================

dff = df.copy()


# =========================================================
# 9. CALCULATE BMI
# =========================================================

# Height in dataset is in centimeters.
# BMI requires height in meters.

dff["height_m"] = dff["height"] / 100


# BMI formula:
# BMI = weight / height²

dff["bmi"] = (
    dff["weight"] /
    (dff["height_m"] ** 2)
)


# =========================================================
# 10. CREATE AGE GROUP
# =========================================================

def get_age_group(age):

    if age <= 18:
        return "Teen"

    elif age <= 45:
        return "Adult"

    else:
        return "Senior"


dff["age_group"] = dff["age"].apply(
    get_age_group
)


# =========================================================
# 11. CREATE LIFESTYLE RISK
# =========================================================

def lifestyle(row):

    if row["smoker"] == "Yes" and row["bmi"] > 30:
        return "high"

    elif row["smoker"] == "Yes" and row["bmi"] > 27:
        return "medium"

    else:
        return "low"


dff["lifestyle_risk"] = dff.apply(
    lifestyle,
    axis=1
)


# =========================================================
# 12. SELECT FEATURES
# =========================================================

X = dff[
    [
        "bmi",
        "age_group",
        "lifestyle_risk",
        "city_tier",
        "income_lpa",
        "occupation"
    ]
]


# =========================================================
# 13. SELECT TARGET
# =========================================================

y = dff[
    "insurance_premium_category"
]


# =========================================================
# 14. DISPLAY FEATURES AND TARGET
# =========================================================

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# =========================================================
# 15. DEFINE CATEGORICAL FEATURES
# =========================================================

categorical_features = [
    "age_group",
    "lifestyle_risk",
    "occupation",
    "city_tier"
]


# =========================================================
# 16. DEFINE NUMERIC FEATURES
# =========================================================

numeric_features = [
    "bmi",
    "income_lpa"
]


# =========================================================
# 17. CREATE PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)


# =========================================================
# 18. CREATE MACHINE LEARNING PIPELINE
# =========================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                random_state=42
            )
        )
    ]
)


# =========================================================
# 19. SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)


# =========================================================
# 20. TRAIN MODEL
# =========================================================

pipeline.fit(
    X_train,
    y_train
)


print("\nModel training completed!")


# =========================================================
# 21. MAKE PREDICTIONS
# =========================================================

y_pred = pipeline.predict(
    X_test
)


# =========================================================
# 22. CALCULATE ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nAccuracy:")
print(accuracy)


# =========================================================
# 23. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================================================
# 24. SAVE MODEL
# =========================================================

pickle_model_path = "model.pkl"


with open(
    pickle_model_path,
    "wb"
) as f:

    pickle.dump(
        pipeline,
        f
    )


print("\nModel saved successfully!")
print("File name: model.pkl")

