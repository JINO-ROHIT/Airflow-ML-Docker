import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    xgb_classifier = xgb.XGBClassifier(random_state=42)

    param_grid = {
        'n_estimators': [500, 750, 1000],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5],
        'min_child_weight': [1, 3, 5]
    }

    grid_search = GridSearchCV(xgb_classifier, param_grid, cv=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_xgb_model = grid_search.best_estimator_
    return best_xgb_model

def preprocess_data(**kwargs):
    
    task_instance = kwargs['ti']
    
    data = task_instance.xcom_pull(task_ids="read_csv")

    data = data.drop(columns=['number'])
    data['type2'].fillna(data['type1'], inplace=True)

    label_encoder = LabelEncoder()
    for feature in ['type1', 'type2', 'legendary', 'name']:
        data[feature] = label_encoder.fit_transform(data[feature])

    X = data.drop(columns=['legendary'])
    y = data['legendary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train.to_dict(), X_test.to_dict(), y_train.to_dict(), y_test.to_dict()


def tune( **kwargs):
    
    task_instance = kwargs['ti']
    
    X_train_dict, _, y_train_dict, _ = task_instance.xcom_pull(task_ids="preprocess_data")
    
    X_train = pd.DataFrame.from_dict(X_train_dict)
    y_train = pd.Series(y_train_dict)
    
    best_model = train_xgboost_model(X_train, y_train)
    
    model_filepath = f"{model_path}/tuned_xgboost.pkl"
    joblib.dump(best_model, model_filepath)
    
    return model_filepath


def test_model(**kwargs):
    
    task_instance = kwargs['ti']
    
    model_filepath = task_instance.xcom_pull(task_ids="tune")
    print(model_filepath)

    model = joblib.load(model_filepath)
    
    _, X_test_dict, _, y_test_dict = task_instance.xcom_pull(task_ids="preprocess_data")
    
    X_test = pd.DataFrame.from_dict(X_test_dict)
    y_test = pd.Series(y_test_dict)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Tuned Xgboost Model Accuracy:", accuracy_score(y_test, y_pred))
    return accuracy


default_args = {
    "owner": "jino",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "start_date": datetime(2023, 12, 1),
}


dag = DAG(
    "pokemonclassification",
    default_args=default_args,
    description="A pipeline to read CSV, preprocess data, tune a model and test on the validation set",
    schedule_interval=timedelta(days=1),
    catchup=False,
)

model_path = "/opt/airflow/model" 
file_path = "/opt/airflow/data/Pokemon.csv"

t1 = PythonOperator(
    task_id="read_csv",
    python_callable=read_csv,
    op_args=[file_path],
    dag=dag,
)

t2 = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id="tune",
    python_callable=tune,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id="test_model",
    python_callable=test_model,
    provide_context=True,
    dag=dag,
)


t1 >> t2 >> t3 >> t4