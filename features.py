#Making Feature Engineering of simulation
import pandas as pd
import ta

def compute_technical_indicators(df):
	df = df.copy()
	df["Change"].str.replace("%", " ", regex=False)
	df["Change"] = pd.to_numeric(df["Change"], errors="coerce")
	df["Price"].str.replace(",", " ", regex=False)
	df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
	df["Low"].str.replace(",", " ", regex=False)
	df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
	df["High"].str.replace(",", " ", regex=False)
	df["High"] = pd.to_numeric(df["High"], errors="coerce")
	df["Open"].str.replace(",", " ", regex=False)
	df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
	df['Volume'] = df['Volume'].replace(',', '', regex=True)
	df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

	#renaming the dataset columns for TA Compatibility
	df = df.rename(columns={
		"Price": "Close",
		"Open": "Open",
		"High": "High",
		"Low": "Low",
		"Change": "Change"
	})

	#Ensure Datetime
	df["Date"] = pd.to_datetime(df["Date"])
	df = df.sort_values("Date")


	#Compute Technical Indicators
	df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
	df["MACD"] = ta.trend.MACD(df["Close"]).macd_diff()
	df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
	df["CCI"] = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"]).cci()
	df["AO"] = ta.momentum.AwesomeOscillatorIndicator(df["High"], df["Low"]).awesome_oscillator()
	df["BB_width"] = ta.volatility.BollingerBands(df["Close"]).bollinger_wband()
	df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
	df["MFI"] = ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).money_flow_index()
	df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()

	#Then, Target Features
	df["Return"] = df["Close"].pct_change()
	df["Mu"] = df["Return"].rolling(10).mean()
	df["Sigma"] = df["Return"].rolling(10).std()

	df = df.drop(columns=["Stock Name", "Code", "Date", "Change"], errors="ignore")
	df = df.fillna(0)
	return df.reset_index(drop=True)


