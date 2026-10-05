# Insurance Premium Category Prediction

## Project Overview

Insurance Premium Category Prediction is a Machine Learning project that predicts an insurance premium category based on user information such as age, weight, height, income, smoking status, city, and occupation.

The project uses Python, Pandas, Scikit-learn, FastAPI, and Streamlit.

The Machine Learning model is trained using a Random Forest Classifier. FastAPI is used to create a backend API for prediction, while Streamlit provides an interactive frontend where users can enter their details and receive the predicted insurance premium category.

The project predicts three categories:

* Low
* Medium
* High

## Project Workflow

```text
User
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ↓
Feature Engineering
  ↓
Trained Machine Learning Pipeline
  ↓
Random Forest Classifier
  ↓
Prediction
  ↓
Streamlit Frontend
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Pickle
* FastAPI
* Pydantic
* Uvicorn
* Streamlit
* Requests

## Machine Learning Model

The project uses a Random Forest Classifier for predicting the insurance premium category.

The model is trained using the following features:

* BMI
* Age Group
* Lifestyle Risk
* City Tier
* Income LPA
* Occupation

The target variable is:

```text
insurance_premium_category
```

The model predicts:

```text
Low
Medium
High
```

## Feature Engineering

Several features are created from the original dataset.

### BMI

BMI is calculated using weight and height.

```text
BMI = Weight / Height²
```

Since height is stored in centimeters, it is first converted to meters.

```python
height_m = height / 100
bmi = weight / (height_m ** 2)
```

### Age Group

Age is converted into three groups:

```text
Age <= 18  → Teen
Age <= 45  → Adult
Age > 45   → Senior
```

### Lifestyle Risk

Lifestyle risk is calculated using smoking status and BMI.

```text
Smoker + BMI > 30 → High Risk
Smoker + BMI > 27 → Medium Risk
Otherwise          → Low Risk
```

### City Tier

Cities are categorized into different tiers.

```text
Tier 1
Tier 2
```

This information is then provided to the Machine Learning model.

## Machine Learning Pipeline

The project uses a Scikit-learn Pipeline.

Categorical features:

```text
age_group
lifestyle_risk
occupation
city_tier
```

Numerical features:

```text
bmi
income_lpa
```

Categorical features are converted into numerical form using OneHotEncoder.

The final model is:

```text
RandomForestClassifier
```

The complete preprocessing and trained model are stored together in:

```text
model.pkl
```

## Project Files

```text
Insurance-Premium-Prediction/
│
├── ml.py
├── app.py
├── frontend.py
├── model.pkl
├── insurance.csv
└── README.md
```

### ml.py

This is the Machine Learning training file.

It performs the following tasks:

1. Loads the insurance dataset.
2. Cleans the data.
3. Converts columns into appropriate data types.
4. Calculates BMI.
5. Creates age groups.
6. Creates lifestyle risk.
7. Selects features and target.
8. Splits the dataset into training and testing data.
9. Creates preprocessing steps.
10. Trains the Random Forest model.
11. Evaluates the model.
12. Saves the complete trained pipeline as `model.pkl`.

### app.py

This is the FastAPI backend.

It:

1. Loads `model.pkl`.
2. Defines the input data using Pydantic.
3. Validates user input.
4. Calculates BMI.
5. Calculates age group.
6. Calculates lifestyle risk.
7. Determines city tier.
8. Sends the processed data to the ML model.
9. Returns the predicted insurance premium category.

The main API endpoint is:

```text
POST /predict
```

FastAPI automatically provides interactive API documentation at:

```text
/docs
```

### frontend.py

This is the Streamlit frontend.

It provides an interactive interface where users can enter:

* Age
* Weight
* Height
* Income
* Smoking status
* City
* Occupation

When the user clicks the prediction button, Streamlit sends the data to the FastAPI backend.

The prediction returned by FastAPI is displayed on the Streamlit webpage.

### model.pkl

This file contains the trained Machine Learning pipeline.

It includes the preprocessing steps and trained Random Forest model.

The file is created by `ml.py` using Python's Pickle library.

Example:

```python
pickle.dump(pipeline, f)
```

The FastAPI application later loads it using:

```python
pickle.load(f)
```

### insurance.csv

This is the dataset used to train the Machine Learning model.

It contains information such as:

* Age
* Weight
* Height
* Income LPA
* Smoker
* City Tier
* Occupation
* Insurance Premium Category

## How the Files Are Connected

The files work together in the following order:

```text
insurance.csv
     ↓
   ml.py
     ↓
 model.pkl
     ↓
   app.py
     ↓
 frontend.py
```

### Step 1: Dataset

`insurance.csv` contains the original training data.

### Step 2: Training

`ml.py` reads the dataset and trains the Machine Learning model.

### Step 3: Model Saving

After training, the pipeline is saved as:

```text
model.pkl
```

### Step 4: FastAPI

`app.py` loads `model.pkl` and creates the `/predict` API.

### Step 5: Streamlit

`frontend.py` sends user input to the FastAPI `/predict` endpoint.

### Step 6: Prediction

FastAPI sends the processed input to the trained model and returns:

```json
{
    "predicted_category": "Low"
}
```

Streamlit displays the result to the user.

## Installation

Create a virtual environment:

```bash
python -m venv myvenv
```

Activate the virtual environment on Windows:

```bash
myvenv\Scripts\activate
```

Install the required libraries:

```bash
python -m pip install pandas numpy scikit-learn fastapi uvicorn streamlit requests
```

## Running the Project

### Step 1: Train the Model

Run:

```bash
python ml.py
```

This will train the model and create:

```text
model.pkl
```

### Step 2: Start FastAPI

Run:

```bash
python -m uvicorn app:app --reload
```

FastAPI will run at:

```text
http://127.0.0.1:8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Step 3: Start Streamlit

Open another terminal and run:

```bash
python -m streamlit run frontend.py
```

Streamlit will open the project in your browser.

## API Example

The `/predict` endpoint accepts data similar to:

```json
{
    "age": 30,
    "weight": 65,
    "height": 170,
    "income_lpa": 10,
    "smoker": "No",
    "city": "Mumbai",
    "occupation": "Student"
}
```

Example response:

```json
{
    "predicted_category": "Low"
}
```

## Important Data Flow

The frontend does not directly load `model.pkl`.

Instead:

```text
frontend.py
     ↓
HTTP POST Request
     ↓
app.py
     ↓
model.pkl
     ↓
Prediction
     ↓
app.py
     ↓
JSON Response
     ↓
frontend.py
```

This separation makes the project similar to a real-world Machine Learning application where the frontend and backend communicate through an API.

## Why FastAPI?

FastAPI is used to expose the Machine Learning model as an API.

Advantages include:

* Fast performance
* Easy API development
* Automatic API documentation
* Pydantic data validation
* Easy integration with Machine Learning models
* Suitable for production-style ML applications

## Why Streamlit?

Streamlit is used to create a simple and interactive frontend without requiring a separate HTML/CSS/JavaScript application.

It allows users to enter their information and view the prediction easily.

## Key Learning Concepts

This project demonstrates:

* Data Cleaning
* Exploratory Data Preparation
* Feature Engineering
* BMI Calculation
* Conditional Feature Creation
* Categorical Data Encoding
* One-Hot Encoding
* Train-Test Split
* Random Forest Classification
* Machine Learning Pipeline
* Model Serialization using Pickle
* REST API Development
* FastAPI
* Pydantic Validation
* HTTP POST Requests
* Streamlit Frontend
* Frontend-Backend Integration

## Future Improvements

Possible improvements include:

* Add model accuracy and classification metrics to the dashboard.
* Add prediction probabilities.
* Improve city and occupation validation.
* Add more insurance-related features.
* Use a larger real-world dataset.
* Add authentication to the API.
* Deploy FastAPI to a cloud server.
* Deploy Streamlit separately.
* Add Docker support.
* Add logging and error monitoring.

## Conclusion

This project demonstrates a complete Machine Learning application workflow.

The Machine Learning model is trained using `ml.py`, saved as `model.pkl`, exposed through a FastAPI backend using `app.py`, and connected to an interactive Streamlit frontend using `frontend.py`.

The project shows how a trained Machine Learning model can be converted into a practical application that accepts real-time user input and returns predictions through an API.

## Author

Aakansha Saxena

Machine Learning | Python | Data Science | FastAPI | Streamlit
