import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

np.random.seed(42)
n_samples = 1000 

data = {
    'hydration_level': np.random.randint(0, 11, n_samples),
    'physical_activity': np.random.randint(0, 11, n_samples),
    'temperature_exposure': np.random.randint(0, 11, n_samples),
    'sleep_quality': np.random.randint(0, 11, n_samples),
    'medication_adherence': np.random.randint(0, 11, n_samples),
    'pain': np.random.randint(0, 11, n_samples),
    'weakness': np.random.randint(0, 11, n_samples),
    'headache': np.random.randint(0, 11, n_samples),
    'dizziness': np.random.randint(0, 11, n_samples),
    'jaundice': np.random.randint(0, 11, n_samples),
    'swelling': np.random.randint(0, 11, n_samples),
    'shortness_of_breath': np.random.randint(0, 11, n_samples),
    'fever': np.random.randint(0, 11, n_samples),
    'chest_pain': np.random.randint(0, 11, n_samples),
    'vision_problems': np.random.randint(0, 11, n_samples),
    'erection': np.random.randint(0, 11, n_samples),
    'stroke_symptoms': np.random.randint(0, 11, n_samples),
    'crisis_risk': np.random.randint(0, 2, n_samples) 
}

df = pd.DataFrame(data)
print(df.head()) 

df.fillna(df.mean(), inplace=True)


X = df.drop('crisis_risk', axis=1)
y = df['crisis_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_

y_pred_best = best_rf_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Model Accuracy after tuning: {accuracy_best * 100:.2f}%")

joblib.dump(rf_model, 'random_forest_crisis_model.pkl')

