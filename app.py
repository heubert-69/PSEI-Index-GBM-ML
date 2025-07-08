import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
from features import compute_technical_indicators

# Load models
mu_model = joblib.load("models/model_mu.pkl")
sigma_model = joblib.load("models/model_sigma.pkl")

def predict_from_csv(filepath):
    df = pd.read_csv(filepath)
    df = compute_technical_indicators(df)

    features = ['RSI', 'MACD', 'ADX', 'CCI', 'AO', 'BB_width', 'ATR', 'MFI', 'OBV']
    X = df[features]

    mu_preds = mu_model.predict(X)
    sigma_preds = sigma_model.predict(X)

    return mu_preds, sigma_preds, df['Date'][-len(mu_preds):] if 'Date' in df.columns else None

def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filepath:
        return
    try:
        mu_preds, sigma_preds, dates = predict_from_csv(filepath)

        result_text.delete("1.0", tk.END)
        for i in range(len(mu_preds)):
            line = f"{dates.iloc[i] if dates is not None else i}: Drift (mu) = {mu_preds[i]:.5f}, Volatility (sigma) = {sigma_preds[i]:.5f}\n"
            result_text.insert(tk.END, line)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI setup
root = tk.Tk()
root.title("PSEI GBM Simulator")
root.geometry("700x500")

frame = tk.Frame(root)
frame.pack(pady=20)

upload_btn = tk.Button(frame, text="Upload CSV & Predict", command=open_file)
upload_btn.pack()

result_text = tk.Text(root, wrap=tk.WORD, height=25)
result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

root.mainloop()

