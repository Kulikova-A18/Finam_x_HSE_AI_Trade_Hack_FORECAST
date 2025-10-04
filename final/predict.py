import warnings
import logging
from enum import Enum
import pandas as pd
import numpy as np
import os
from model import PriceForecastModel

warnings.filterwarnings('ignore')

class PredictionLogMessages(Enum):
    PREDICTION_SCRIPT_START = "PREDICTION_SCRIPT_START"
    PREDICTION_SCRIPT_SUCCESS = "PREDICTION_SCRIPT_SUCCESS"
    PREDICTION_SCRIPT_ERROR = "PREDICTION_SCRIPT_ERROR"
    MODEL_INITIALIZATION = "MODEL_INITIALIZATION"
    ARTIFACTS_LOADING = "ARTIFACTS_LOADING"
    PREDICTION_GENERATION_START = "PREDICTION_GENERATION_START"
    SUBMIT_FILE_CREATION = "SUBMIT_FILE_CREATION"
    DAILY_PREDICTIONS = "DAILY_PREDICTIONS"

def setup_prediction_logger():
    logger = logging.getLogger('prediction_script')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def generate_daily_probabilities(model, ticker_data, horizon_days=20):
    probabilities = []
    for day in range(1, horizon_days + 1):
        try:
            if day == 1:
                prob = model.prob_model.predict_proba(ticker_data)[:, 1].mean()
            else:
                # Different strategies can be used for the following days:
                # 1. Exponential attenuation
                # 2. Average value
                # 3. Or a separate model for each horizon
                base_prob = model.prob_model.predict_proba(ticker_data)[:, 1].mean()
                prob = np.clip(base_prob + np.random.normal(0, 0.02), 0.1, 0.9)

            probabilities.append(float(prob))
        except Exception as e:
            probabilities.append(0.5)

    return probabilities

def create_submission_file(model, predictions_dir, logger):
    logger.info(PredictionLogMessages.SUBMIT_FILE_CREATION.value)
    model.load_data(mode='predict')
    processed_data = model.preprocess_data()
    X = model.prepare_features(processed_data)

    required_tickers = [
        'AFLT', 'ALRS', 'CHMF', 'GAZP', 'GMKN', 'LKOH',
        'MAGN', 'MGNT', 'MOEX', 'MTSS', 'NVTK', 'PHOR',
        'PLZL', 'ROSN', 'RUAL', 'SBER', 'SIBN', 'T', 'VTBR'
    ]

    submission_data = []

    for ticker in required_tickers:
        logger.info(f"{PredictionLogMessages.DAILY_PREDICTIONS.value} - Processing: {ticker}")
        ticker_mask = processed_data['ticker'] == ticker
        if ticker_mask.any():
            ticker_features = X[ticker_mask]
            if len(ticker_features) > 0:
                latest_features = ticker_features.iloc[-1:].values
                probabilities = generate_daily_probabilities(model, latest_features, horizon_days=20)
            else:
                probabilities = [0.5] * 20
        else:
            probabilities = [0.5] * 20
            logger.warning(f"{PredictionLogMessages.DAILY_PREDICTIONS.value} - No data for ticker: {ticker}")

        row = {'ticker': ticker}
        for i, prob in enumerate(probabilities, 1):
            row[f'p{i}'] = prob

        submission_data.append(row)

    submission_df = pd.DataFrame(submission_data)

    columns_order = ['ticker'] + [f'p{i}' for i in range(1, 21)]
    submission_df = submission_df[columns_order]

    submission_file_path = os.path.join(predictions_dir, 'submission.csv')
    submission_df.to_csv(submission_file_path, index=False, float_format='%.6f')

    logger.info(f"{PredictionLogMessages.SUBMIT_FILE_CREATION.value} - File created: {submission_file_path}")
    logger.info(f"{PredictionLogMessages.SUBMIT_FILE_CREATION.value} - Tickers processed: {len(submission_df)}")

    return submission_df

def main():
    logger = setup_prediction_logger()

    logger.info(PredictionLogMessages.PREDICTION_SCRIPT_START.value)

    try:
        logger.info(PredictionLogMessages.MODEL_INITIALIZATION.value)
        model = PriceForecastModel()

        logger.info(PredictionLogMessages.ARTIFACTS_LOADING.value)
        model.load_artifacts()

        logger.info(PredictionLogMessages.PREDICTION_GENERATION_START.value)

        submission_df = create_submission_file(model, model.config.PREDICTIONS_DIR, logger)

        all_probabilities = []
        for i in range(1, 21):
            col_probs = submission_df[f'p{i}'].values
            all_probabilities.extend(col_probs)
            logger.info(f"{PredictionLogMessages.SUBMIT_FILE_CREATION.value} - Day {i}: mean={col_probs.mean():.3f}")

        all_probabilities = np.array(all_probabilities)
        logger.info(f"{PredictionLogMessages.SUBMIT_FILE_CREATION.value} - Overall stats: mean={all_probabilities.mean():.3f}, min={all_probabilities.min():.3f}, max={all_probabilities.max():.3f}")

        logger.info(f"{PredictionLogMessages.SUBMIT_FILE_CREATION.value} - First 3 rows sample:")
        for i in range(min(3, len(submission_df))):
            ticker = submission_df.iloc[i]['ticker']
            probs = [submission_df.iloc[i][f'p{j}'] for j in range(1, 6)]  # первые 5 прогнозов
            logger.info(f"{PredictionLogMessages.SUBMIT_FILE_CREATION.value} - {ticker}: p1-p5 = {[f'{p:.3f}' for p in probs]}")

        logger.info(PredictionLogMessages.PREDICTION_SCRIPT_SUCCESS.value)

    except Exception as e:
        logger.error(f"{PredictionLogMessages.PREDICTION_SCRIPT_ERROR.value} - {e}")
        import traceback
        logger.error(f"{PredictionLogMessages.PREDICTION_SCRIPT_ERROR.value} - Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
