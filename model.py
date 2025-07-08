#training Models for the Machine Learning Side
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
from features import compute_technical_indicators
import numpy as np

df = pd.read_csv("data/stocks.csv")
df = compute_technical_indicators(df)

features = ["RSI", "MACD", "ADX", "CCI", "AO", "BB_width", "ATR", "MFI", "OBV"]
X = df[features]
y_mu = df["Mu"]
y_sigma = df["Sigma"]

#Splitting of the data
X_train, X_test, Y_mu_train, Y_mu_test = train_test_split(X, y_mu, test_size=0.2, random_state=42)
X_train_sigma, X_test_sigma, Y_train_sigma, Y_test_sigma = train_test_split(X, y_sigma, test_size=0.2, random_state=42)

#Pipeline to avoid Data Leakage
pipeline_mu = Pipeline([
	("scaler", StandardScaler()),
	("model", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42))
])

pipeline_sigma = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42))
])

#Let The Training begin
pipeline_mu.fit(X_train, Y_mu_train)
pipeline_sigma.fit(X_train_sigma, Y_train_sigma)


#Cross Validation
cv_scores_mu = cross_val_score(pipeline_mu, X, y_mu, cv=5, scoring="r2")
cv_scores_sigma = cross_val_score(pipeline_sigma, X, y_sigma, cv=5, scoring="r2")
print(f"R^2 Score of Mu: {np.mean(cv_scores_mu)}")
print(f"R^2 Score of Sigma: {np.mean(cv_scores_sigma)}")


#Model Evaluation
def EvaluateModel(name, model, X_test, y_true):
	y_pred = model.predict(X_test)
	print(f"{name}'s Evaluation:\n")
	print(f"MSE: {mean_squared_error(y_true, y_pred)}")
	print(f"MAE: {mean_absolute_error(y_true, y_pred)}")
	print(f"R^2 Score: {r2_score(y_true, y_pred)}")

EvaluateModel("Mu", pipeline_mu, X_test, Y_mu_test)
EvaluateModel("Sigma", pipeline_sigma, X_test_sigma, Y_test_sigma)


#Then, We dump the models into a pickle file
dump(pipeline_mu, "models/model_mu.pkl")
dump(pipeline_sigma, "models/model_sigma.pkl")
