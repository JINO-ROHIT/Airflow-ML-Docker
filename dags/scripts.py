from typing import Dict, Tuple

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocesses the Pokemon data."""
    data = data.drop(columns=['number'])
    data['type2'].fillna(data['type1'], inplace=True)

    label_encoder = LabelEncoder()
    for feature in ['type1', 'type2', 'legendary', 'name']:
        data[feature] = label_encoder.fit_transform(data[feature])

    X = data.drop(columns=['legendary'])
    y = data['legendary']
    return X, y


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """Trains an XGBoost model."""
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


def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluates the trained model."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def save_model(model: xgb.XGBClassifier, filepath: str) -> None:
    """Saves the trained model."""
    joblib.dump(model, filepath)


def load_model(filepath: str) -> xgb.XGBClassifier:
    """Loads a saved model."""
    return joblib.load(filepath)


if __name__ == "__main__":
    pokemon_data = pd.read_csv("data/Pokemon.csv")

    X, y = preprocess_data(pokemon_data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    best_model = train_xgboost_model(X_train, y_train)

    # Evaluate the model
    evaluation_results = evaluate_model(best_model, X_test, y_test)
    print("Evaluation Results:")
    print("Accuracy:", evaluation_results["accuracy"])
    print("Classification Report:")
    print(evaluation_results["classification_report"])

    # Save the model
    model_filepath = 'model/xgboost.pkl'
    save_model(best_model, model_filepath)

    # Load the model
    loaded_model = load_model(model_filepath)
