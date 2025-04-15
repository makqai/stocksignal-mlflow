import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from ta.momentum import RSIIndicator
import pickle
from dotenv import load_dotenv
import os
import warnings
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import json
import argparse

warnings.filterwarnings("ignore")

# Parse command-line arguments for flexibility
parser = argparse.ArgumentParser(description="MLflow Pipeline: Train, Evaluate, Infer")
parser.add_argument("--data-path", type=str, default=os.getenv("DATA_PATH", "data/NetGEX_AbsGEX_EPS(AAPL).pickle"))
parser.add_argument("--model-path", type=str, default=os.getenv("MODEL_PATH", "model/NetGEX_AbsGEX_EPS(AAPL).pkl"))
parser.add_argument("--scaler-path", type=str, default=os.getenv("SCALER_PATH", "scalar/NetGEX_AbsGEX_EPS(AAPL).pkl"))
parser.add_argument("--prediction-horizon", type=int, default=5)
parser.add_argument("--price-change-threshold", type=float, default=0.5)
parser.add_argument("--train-split-ratio", type=float, default=0.8)
parser.add_argument("--n-estimators", type=int, default=50)
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument("--class-weight", type=str, default="balanced")
parser.add_argument("--initial-capital", type=float, default=10000)
parser.add_argument("--transaction-cost", type=float, default=0.01)
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = args.data_path or os.getenv("DATA_PATH")
MODEL_PATH = args.model_path or os.getenv("MODEL_PATH")
SCALER_PATH = args.scaler_path or os.getenv("SCALER_PATH")
PREDICTION_HORIZON = args.prediction_horizon or int(os.getenv("PREDICTION_HORIZON"))
PRICE_CHANGE_THRESHOLD = args.price_change_threshold or float(os.getenv("PRICE_CHANGE_THRESHOLD"))
TRAIN_SPLIT_RATIO = args.train_split_ratio or float(os.getenv("TRAIN_SPLIT_RATIO"))
FEATURES = os.getenv("FEATURES").split(",")
MOVING_AVERAGE_WINDOWS = [int(x) for x in os.getenv("MOVING_AVERAGE_WINDOWS").split(",")]
RSI_WINDOW = int(os.getenv("RSI_WINDOW"))
LAG_PERIODS = [int(x) for x in os.getenv("LAG_PERIODS").split(",")]
N_ESTIMATORS = args.n_estimators or int(os.getenv("N_ESTIMATORS"))
RANDOM_STATE = args.random_state or int(os.getenv("RANDOM_STATE"))
CLASS_WEIGHT = args.class_weight or os.getenv("CLASS_WEIGHT")
INITIAL_CAPITAL = args.initial_capital or float(os.getenv("INITIAL_CAPITAL"))
TRANSACTION_COST = args.transaction_cost or float(os.getenv("TRANSACTION_COST"))

# Start MLflow run
with mlflow.start_run(run_name="stock_prediction_pipeline"):
    # Log parameters
    mlflow.log_param("data_path", DATA_PATH)
    mlflow.log_param("prediction_horizon", PREDICTION_HORIZON)
    mlflow.log_param("price_change_threshold", PRICE_CHANGE_THRESHOLD)
    mlflow.log_param("train_split_ratio", TRAIN_SPLIT_RATIO)
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("class_weight", CLASS_WEIGHT)
    mlflow.log_param("initial_capital", INITIAL_CAPITAL)
    mlflow.log_param("transaction_cost", TRANSACTION_COST)
    mlflow.log_param("features", ",".join(FEATURES))
    mlflow.log_param("moving_average_windows", MOVING_AVERAGE_WINDOWS)
    mlflow.log_param("rsi_window", RSI_WINDOW)
    mlflow.log_param("lag_periods", LAG_PERIODS)

    # Step 1: Load data
    print("Loading data...")
    df = pd.read_pickle(DATA_PATH)
    print("First few rows:")
    print(df.head())

    # Step 2: Preprocess data
    print("Preprocessing data...")
    df.index = pd.to_datetime(df.index)
    df = df.drop(['volume_abs_gex', 'volume_net_gex'], axis=1)
    df['future_close'] = df['Spot_Close'].shift(-PREDICTION_HORIZON)
    df['price_change'] = df['future_close'] - df['Spot_Close']
    df['signal'] = 0
    df.loc[df['price_change'] > PRICE_CHANGE_THRESHOLD, 'signal'] = 1
    df.loc[df['price_change'] < -PRICE_CHANGE_THRESHOLD, 'signal'] = -1
    df = df.dropna()

    # Step 3: Engineer features
    print("Engineering features...")
    df['price_diff'] = df['Spot_Close'] - df['Spot_Open']
    df['price_range'] = df['Spot_High'] - df['Spot_Low']
    for window in MOVING_AVERAGE_WINDOWS:
        df[f'ma{window}'] = df['Spot_Close'].rolling(window=window).mean()
    df['rsi'] = RSIIndicator(df['Spot_Close'], window=RSI_WINDOW).rsi()
    df['net_gex_diff'] = df['open_net_gex'].diff()
    df['abs_gex_diff'] = df['open_abs_gex'].diff()
    df['eps_diff'] = df['PCT_EPS_1mo_Close'].diff()
    for lag in LAG_PERIODS:
        df[f'lag{lag}_close'] = df['Spot_Close'].shift(lag)
    df = df.dropna()

    # Step 4: Prepare data
    X = df[FEATURES]
    y = df['signal']
    train_size = int(TRAIN_SPLIT_RATIO * len(df))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    test_df = df[train_size:].copy()

    # Step 5: Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_scaled, columns=FEATURES, index=X_train.index)
    X_test_scaled = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test_scaled, columns=FEATURES, index=X_test.index)

    # Step 6: Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT
    )
    model.fit(X_train, y_train)

    # Log model and scaler
    mlflow.sklearn.log_model(model, "random_forest_model", input_example=X_test.iloc[:1])
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact("scaler.pkl")
    os.remove("scaler.pkl")  # Clean up

    # Save to original paths for compatibility
    """ with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f) """

    # Step 7: Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    clf_report = classification_report(y_test, y_pred, target_names=['Sell', 'Hold', 'Buy'], output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Sell', 'Hold', 'Buy']))

    # Log evaluation metrics
    for label in ['Sell', 'Hold', 'Buy']:
        mlflow.log_metric(f"{label}_precision", clf_report[label]['precision'])
        mlflow.log_metric(f"{label}_recall", clf_report[label]['recall'])
        mlflow.log_metric(f"{label}_f1-score", clf_report[label]['f1-score'])
    mlflow.log_metric("accuracy", clf_report['accuracy'])

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    with open("confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f)
    mlflow.log_artifact("confusion_matrix.json")
    os.remove("confusion_matrix.json")

    # Log feature importance
    feature_importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    feature_importance.to_csv("feature_importance.csv")
    mlflow.log_artifact("feature_importance.csv")
    os.remove("feature_importance.csv")

    # Step 8: Inference and backtesting
    print("Running inference and backtest...")
    test_df['predicted_signal'] = y_pred

    capital = INITIAL_CAPITAL
    position = 0
    trades = []

    for i in test_df.index:
        signal = test_df.loc[i, 'predicted_signal']
        price = test_df.loc[i, 'Spot_Close']
        
        if signal == 1 and position == 0:  # Buy
            shares = capital // price
            if shares > 0:
                position = shares
                capital -= shares * price
                capital -= shares * TRANSACTION_COST
                trades.append(('Buy', price, i))
        
        elif signal == -1 and position > 0:  # Sell
            capital += position * price
            capital -= position * TRANSACTION_COST
            trades.append(('Sell', price, i))
            position = 0

    # Close any open position
    if position > 0:
        capital += position * test_df['Spot_Close'].iloc[-1]
        capital -= position * TRANSACTION_COST
        trades.append(('Sell', test_df['Spot_Close'].iloc[-1], test_df.index[-1]))
        position = 0

    final_capital = capital
    returns = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"\nBacktest Results:")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Final Capital: ${final_capital:.2f}")
    print(f"Returns: {returns:.2f}%")
    print(f"Number of trades: {len(trades)}")

    # Log backtest metrics
    mlflow.log_metric("final_capital", final_capital)
    mlflow.log_metric("returns_percent", returns)
    mlflow.log_metric("number_of_trades", len(trades))

    # Step 9: Visualize results
    print("Generating plot...")
    plt.figure(figsize=(14, 7))
    plt.plot(test_df.index, test_df['Spot_Close'], label='Spot Close', color='blue')
    buy_signals = test_df[test_df['predicted_signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals['Spot_Close'], 
               label='Buy Signal', color='green', marker='^', s=100)
    sell_signals = test_df[test_df['predicted_signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals['Spot_Close'], 
               label='Sell Signal', color='red', marker='v', s=100)
    plt.title('AAPL Stock Price with Buy/Sell Signals')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("signals_plot.png")
    mlflow.log_artifact("signals_plot.png")
    os.remove("signals_plot.png")
    print("Pipeline completed!")