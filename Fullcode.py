import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Constants
NUMERICAL_COLS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Load dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['type'] = LabelEncoder().fit_transform(data['type'])
    data[NUMERICAL_COLS] = data[NUMERICAL_COLS].apply(pd.to_numeric, downcast='float')
    reshaped_data, _ = train_test_split(data, train_size=100000, stratify=data['isFraud'], random_state=42)
    return reshaped_data

# Visualization
def plot_transaction_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='isFraud', data=data, palette=['#00C853', '#D50000'], alpha=0.85)
    plt.title('Fraud vs Non-Fraud Transactions', fontsize=16, fontweight='bold')
    plt.xlabel('Fraudulent Transaction', fontsize=14)
    plt.ylabel('Transaction Count', fontsize=14)
    total = len(data)
    for bar in plt.gca().patches:
        count = bar.get_height()
        percentage = f'{(count / total) * 100:.2f}%'
        plt.gca().text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * total,
        percentage, ha='center', fontsize=12, color='black', fontweight='bold')
    plt.tight_layout()
    plt.show()

# Model training and evaluation
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nEvaluation Metrics:")
    print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# FastAPI app setup
app = FastAPI()

class Transaction(BaseModel):
    transaction_type: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

@app.post('/predict')
async def predict(transaction: Transaction):
    input_data = pd.DataFrame([transaction.dict()])
    input_data[NUMERICAL_COLS] = scaler.transform(input_data[NUMERICAL_COLS])
    input_data['step'] = 0
    prediction = model.predict(input_data)[0]
    return {"fraud_prediction": int(prediction)}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <html>
        <head>
            <title>Fraud Detection</title>
        </head>
        <body>
            <h1>Fraud Detection API</h1>
            <form action="/predict" method="post">
                <label for="transaction_type">Transaction Type:</label><br>
                <input type="number" id="transaction_type" name="transaction_type"><br>
                <label for="amount">Amount:</label><br>
                <input type="number" id="amount" name="amount"><br>
                <label for="oldbalanceOrg">Old Balance Origin:</label><br>
                <input type="number" id="oldbalanceOrg" name="oldbalanceOrg"><br>
                <label for="newbalanceOrig">New Balance Origin:</label><br>
                <input type="number" id="newbalanceOrig" name="newbalanceOrig"><br>
                <label for="oldbalanceDest">Old Balance Destination:</label><br>
                <input type="number" id="oldbalanceDest" name="oldbalanceDest"><br>
                <label for="newbalanceDest">New Balance Destination:</label><br>
                <input type="number" id="newbalanceDest" name="newbalanceDest"><br><br>
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == '__main__':
    reshaped_data = load_and_preprocess_data('Data/Fraud.csv')
    X = reshaped_data.drop(['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1)
    y = reshaped_data['isFraud']
    scaler = StandardScaler()
    X[NUMERICAL_COLS] = scaler.fit_transform(X[NUMERICAL_COLS])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    model = LGBMClassifier(device='gpu', random_state=42)
    train_and_evaluate_model(X_train_resampled, y_train_resampled, X_test, y_test, model)
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
