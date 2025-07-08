# 📈 PSEI Index Simulation with GBM and Machine Learning

This project performs financial simulation and prediction on the Philippine Stock Exchange Index (PSEI) using:

- 🧮 Geometric Brownian Motion (GBM) for drift/volatility modeling
- 🤖 Machine Learning with scikit-learn (Random Forest, Pipelines, Cross-validation)
- 📊 Technical indicators from the `ta` library
- 🖼 A GUI made with `tkinter` for interactive scenario testing

---

## 🚀 Features

- Upload PSEI historical data (with Open, High, Low, Close, Volume)
- Automatically generate optimized technical indicators
- Predict drift (μ) and volatility (σ) using ML pipelines
- Simulate stock price paths under multiple scenarios
- View predictions in an easy-to-use desktop GUI
- Export cleaned or simulated data

---

## 📂 Structure
```bash
PSEIGBM/
│
├── app.py # Tkinter GUI app
├── model.py # Trains RandomForest regressors
├── features.py # Adds TA indicators to dataset
├── sample_testing_cleaned.csv
├── models/
│ ├── model_mu.pkl
│ └── model_sigma.pkl
└── data/
└── stocks.csv # PSEI historical dataset
```

---

## 🧠 Tech Stack

- Python 3.8+
- `scikit-learn`, `pandas`, `ta`
- `joblib`, `tkinter`, `matplotlib`

---

## 🛠 Installation

```bash
git clone https://github.com/heubert-69/PSEI-Index-GBM-ML.git
cd PSEI-Index-GBM-ML
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

✅ Usage
```bash
python app.py
```
Then select your CSV, and let the ML-GBM simulator do the rest.

📜 License
Licensed under the MIT License
