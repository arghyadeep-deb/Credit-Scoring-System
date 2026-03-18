# Credit Scoring Prediction System
A Machine Learning-based application that predicts credit decisions using a hybrid scoring pipeline built with classical ML and deep learning models.
The system uses FastAPI for the backend and Streamlit for the frontend, forming an end-to-end ML deployment pipeline.

---

## About the Project
This project demonstrates how a credit scoring model can be deployed as a REST API and consumed through a simple web interface.
It focuses on applicant risk assessment using a hybrid ensemble of a calibrated machine learning model and a neural network, along with human-readable reasoning for each prediction.

The application supports:
- Encoded feature-based prediction
- Raw applicant field prediction
- Batch inference for multiple records
- Plain-language explanations of model decisions

---

## Features
- Credit decision prediction using a hybrid ML pipeline
- FastAPI backend for real-time inference
- Streamlit frontend for interactive applicant scoring
- Support for both encoded and raw input schemas
- Batch prediction endpoint for multiple records
- Human-readable decision reasoning
- SHAP-based top driver extraction when available
- Model and preprocessing artifact loading from saved files
- Clean modular project structure

---

## Tech Stack
- **Language:** Python
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Deep Learning:** PyTorch
- **Explainability:** SHAP, LIME
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy

---

## Project Structure
```text
Credit Scoring/
|
+-- backend/
|   +-- __init__.py
|   +-- inference.py              # Hybrid model loading and inference logic
|   +-- main.py                   # FastAPI backend
|
+-- frontend/
|   +-- app.py                    # Streamlit frontend
|
+-- data/
|   +-- case_study1.xlsx
|   +-- case_study2.xlsx
|
+-- model/
|   +-- CreditScoring.ipynb       # Model training and experimentation notebook
|   +-- artifacts/
|       +-- hybrid_preprocessing.pkl
|       +-- nn_40_state_dict.pt
|       +-- model_metadata.json
|
+-- requirements.txt
+-- .gitattributes
+-- .gitignore
+-- README.md
```

---

## How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the FastAPI backend
```bash
uvicorn backend.main:app --reload
```

#### Backend URL
```text
http://127.0.0.1:8000
```

#### Swagger UI
```text
http://127.0.0.1:8000/docs
```

### Step 3: Run the Streamlit frontend
```bash
streamlit run frontend/app.py
```

### Step 4: Open the Streamlit app
Streamlit will usually start on:

```text
http://localhost:8501
```

---

## API Endpoints

### `GET /health`
Checks whether the backend service is running.

### `GET /schema`
Returns the encoded feature schema, class labels, rejection label, and model blend weights.

### `GET /raw-schema`
Returns the numeric and categorical raw input fields expected by the frontend and raw prediction API.

### `POST /predict`
Predicts the credit decision from an encoded feature dictionary.

### `POST /predict-raw`
Predicts the credit decision from raw applicant fields.

### `POST /predict-batch`
Runs batch inference on multiple records using either `encoded` or `raw` input type.

---

## Model Details
The scoring pipeline uses a hybrid architecture that combines:

- A calibrated ensemble model for structured tabular predictions
- A PyTorch neural network for additional predictive learning
- Weighted probability blending for final class prediction

The model outputs:
- Final decision (`Accepted` or `Rejected`)
- Predicted credit class
- Confidence score
- Class probabilities
- Top contributing drivers
- Plain-language reasoning

According to the saved metadata:
- **Number of features:** 89
- **Classes:** P1, P2, P3, P4
- **Blend weights:** 0.6 ensemble, 0.4 neural network

---

## Dataset
The project includes spreadsheet-based case study datasets used for credit scoring analysis and model experimentation.
These datasets help the model learn patterns related to applicant risk, delinquency, income behavior, and repayment indicators.

---

## Notes
- Model artifacts are stored in `model/artifacts/`.
- `hybrid_preprocessing.pkl` is tracked with Git LFS, so make sure Git LFS is installed before cloning or pulling large artifacts.
- The frontend fetches the raw schema dynamically from the backend using the `BACKEND_URL` environment variable. If not set, it defaults to `http://127.0.0.1:8000`.

---

## Purpose
This project is intended for learning and demonstrating:

- Credit risk prediction using machine learning
- Hybrid model deployment with FastAPI
- Frontend-backend integration with Streamlit
- Batch and real-time inference workflows
- Explainable AI for credit decisions
- End-to-end ML project structuring and deployment
