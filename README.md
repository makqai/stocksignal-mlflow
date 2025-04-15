# StockSignal-MLFlow

StockSignal-MLFlow is a Machine Learning project designed to generate Buy, Sell, and Hold signals for stocks (AAPL, MSFT, and SPY) using minute-by-minute trading data. It leverages features such as price movements, options-related metrics (Gamma Exposure, Expected Price Swing), and technical indicators (RSI) to predict short-term price changes. The project uses a RandomForestClassifier model and integrates MLflow for experiment tracking, model management, and reproducibility.

## Features
- **Data Preprocessing**: Processes minute-by-minute stock data, including price and options metrics.
- **Feature Engineering**: Computes technical indicators (e.g., RSI, moving averages) and lagged features.
- **Model Training**: Trains a RandomForestClassifier to predict price movements (Buy: +1, Hold: 0, Sell: -1).
- **Evaluation**: Generates classification metrics (precision, recall, f1-score, accuracy) and feature importance.
- **Backtesting**: Simulates trading based on predicted signals, calculating returns and number of trades.
- **Visualization**: Plots stock prices with Buy/Sell signals.
- **MLflow Integration**: Tracks parameters, metrics, and artifacts (model, scaler, plots) for reproducibility.

## Requirements
- Python >= 3.12
- Dependencies (listed in `pyproject.toml`):
  - `mlflow>=2.17.0`
  - `pandas>=2.2.3`
  - `numpy>=1.26.4`
  - `scikit-learn>=1.5.2`
  - `python-dotenv>=1.0.1`
  - `ta>=0.11.0`
  - `matplotlib>=3.9.2`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/makqai/stocksignal-mlflow.git
   cd stocksignal-mlflow
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   - If using `pip`:
     ```bash
     pip install .
     ```
   - Or manually install:
     ```bash
     pip install mlflow pandas numpy scikit-learn python-dotenv ta matplotlib
     ```

   - If using Poetry:
     ```bash
     poetry install
     ```

4. **Prepare the `.env` File**:
   - Copy the example `.env` configuration to a `.env` file in the project root:
     ```bash
     cp .env.example .env
     ```
   - Ensure the `.env` file contains:
     ```plaintext
     # File paths
     DATA_PATH=data/NetGEX_AbsGEX_EPS(XXXX).pickle
     MODEL_PATH=model/NetGEX_AbsGEX_EPS(XXXX).pkl
     SCALER_PATH=scalar/NetGEX_AbsGEX_EPS(XXXX).pkl
     # Data preprocessing
     PREDICTION_HORIZON=5
     PRICE_CHANGE_THRESHOLD=0.5
     TRAIN_SPLIT_RATIO=0.8
     # Features
     FEATURES=Spot_Close,Spot_Open,Spot_High,Spot_Low,price_diff,price_range,ma5,ma10,rsi,open_net_gex,open_abs_gex,net_gex_diff,abs_gex_diff,PCT_EPS_1mo_Close,eps_diff,lag1_close,lag2_close
     MOVING_AVERAGE_WINDOWS=5,10
     RSI_WINDOW=14
     LAG_PERIODS=1,2
     # Model parameters
     N_ESTIMATORS=50
     RANDOM_STATE=42
     CLASS_WEIGHT=balanced
     # Backtesting
     INITIAL_CAPITAL=10000
     TRANSACTION_COST=0.01
     ```
   - Update `DATA_PATH` to point to your stock data file (e.g., `data/NetGEX_AbsGEX_EPS(XXXX).pickle`).

5. **Prepare Data**:
   - Ensure the stock data file (e.g., `NetGEX_AbsGEX_EPS(XXXX).pickle`) is available in the `data/` directory.
   - The data should include columns like `Spot_Close`, `Spot_Open`, `Spot_High`, `Spot_Low`, `open_net_gex`, `open_abs_gex`, and `PCT_EPS_1mo_Close`.

## Usage

1. **Run the Pipeline**:
   - Execute the main script to train, evaluate, and backtest the model:
     ```bash
     python stocksignal-pipeline.py
     ```
   - Override default parameters if needed:
     ```bash
     python stocksignal-pipeline.py --n-estimators 100 --initial-capital 20000
     ```

2. **View MLflow Results**:
   - Start the MLflow UI:
     ```bash
     mlflow ui
     ```
   - Open `http://localhost:5000` in a browser to view:
     - Parameters (e.g., `prediction_horizon`, `n_estimators`).
     - Metrics (e.g., classification accuracy, backtest returns).
     - Artifacts (e.g., model, scaler, confusion matrix, signals plot).

3. **Outputs**:
   - **Console**: Displays data preview, classification report, confusion matrix, feature importance, backtest results, and a plot.
   - **Files**:
     - Model saved to `MODEL_PATH` (e.g., `model/NetGEX_AbsGEX_EPS(XXXX).pkl`).
     - Scaler saved to `SCALER_PATH` (e.g., `scalar/NetGEX_AbsGEX_EPS(XXXX).pkl`).
   - **MLflow Artifacts**:
     - RandomForest model (`random_forest_model`).
     - Scaler (`scaler.pkl`).
     - Confusion matrix (`confusion_matrix.json`).
     - Feature importance (`feature_importance.csv`).
     - Signals plot (`signals_plot.png`).

## Project Structure
```
stocksignal-mlflow/
├── data/                     # Stock data files (e.g., NetGEX_AbsGEX_EPS(XXXX).pickle)
├── .env                      # Environment variables
├── pyproject.toml            # Project metadata and dependencies
├── stocksignal-pipeline.py   # Main script for training, evaluation, and backtesting
├── README.md                 # This file
└── mlruns/                   # MLflow experiment tracking (generated after running)
```

## Configuration
The `.env` file controls the pipeline settings:
- **Data Paths**: `DATA_PATH`, `MODEL_PATH`, `SCALER_PATH`.
- **Preprocessing**: `PREDICTION_HORIZON`, `PRICE_CHANGE_THRESHOLD`, `TRAIN_SPLIT_RATIO`.
- **Features**: `FEATURES`, `MOVING_AVERAGE_WINDOWS`, `RSI_WINDOW`, `LAG_PERIODS`.
- **Model**: `N_ESTIMATORS`, `RANDOM_STATE`, `CLASS_WEIGHT`.
- **Backtesting**: `INITIAL_CAPITAL`, `TRANSACTION_COST`.

Override these via command-line arguments for experimentation:
```bash
python stocksignal-pipeline.py --prediction-horizon 10 --price-change-threshold 0.7
```

## Contributing
- Report issues or suggest features via GitHub Issues.
- Submit pull requests with improvements or bug fixes.
- Ensure code follows PEP 8 and includes tests where applicable.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Built with [MLflow](https://mlflow.org/) for experiment tracking.
- Uses [scikit-learn](https://scikit-learn.org/) for machine learning.
- Technical indicators powered by [ta](https://github.com/bukosabino/ta).