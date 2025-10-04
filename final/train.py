import warnings
import logging
from enum import Enum
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from model import PriceForecastModel
from config import Config

warnings.filterwarnings('ignore')

class TrainingLogMessages(Enum):
    TRAINING_SCRIPT_START = "TRAINING_SCRIPT_START"
    TRAINING_SCRIPT_SUCCESS = "TRAINING_SCRIPT_SUCCESS"
    TRAINING_SCRIPT_ERROR = "TRAINING_SCRIPT_ERROR"
    MODEL_INITIALIZATION = "MODEL_INITIALIZATION"
    TRAINING_PROCESS_START = "TRAINING_PROCESS_START"
    ARTIFACTS_SAVING = "ARTIFACTS_SAVING"
    VALIDATION_PREDICTIONS_START = "VALIDATION_PREDICTIONS_START"
    PREDICTION_HORIZON_START = "PREDICTION_HORIZON_START"
    FINAL_PREDICTIONS_SAVED = "FINAL_PREDICTIONS_SAVED"
    PREDICTION_STATISTICS = "PREDICTION_STATISTICS"

def setup_training_logger():
    logger = logging.getLogger('training_script')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def main():
    logger = setup_training_logger()

    logger.info(TrainingLogMessages.TRAINING_SCRIPT_START.value)

    try:
        logger.info(TrainingLogMessages.MODEL_INITIALIZATION.value)
        model = PriceForecastModel()

        logger.info(TrainingLogMessages.TRAINING_PROCESS_START.value)
        model.train()

        logger.info(TrainingLogMessages.ARTIFACTS_SAVING.value)
        model.save_artifacts()

        logger.info(TrainingLogMessages.VALIDATION_PREDICTIONS_START.value)

        logger.info(f"{TrainingLogMessages.PREDICTION_HORIZON_START.value} - 1 day")
        pred_1d = model.predict(k_days=1)

        logger.info(f"{TrainingLogMessages.PREDICTION_HORIZON_START.value} - 20 days")
        pred_20d = model.predict(k_days=20)

        final_predictions = pred_1d.merge(pred_20d, on='ticker', how='outer')
        final_predictions_path = model.config.PREDICTIONS_DIR + 'summit.csv'
        final_predictions.to_csv(final_predictions_path, index=False)

        logger.info(f"{TrainingLogMessages.FINAL_PREDICTIONS_SAVED.value} - Path: {final_predictions_path}")
        logger.info(f"{TrainingLogMessages.FINAL_PREDICTIONS_SAVED.value} - Columns: {list(final_predictions.columns)}")

        logger.info(TrainingLogMessages.PREDICTION_STATISTICS.value)
        logger.info(f"{TrainingLogMessages.PREDICTION_STATISTICS.value} - Tickers predicted: {len(final_predictions)}")
        logger.info(f"{TrainingLogMessages.PREDICTION_STATISTICS.value} - 1-day return range: [{final_predictions['return_1d'].min():.4f}, {final_predictions['return_1d'].max():.4f}]")
        logger.info(f"{TrainingLogMessages.PREDICTION_STATISTICS.value} - 1-day probability range: [{final_predictions['up_prob_1d'].min():.3f}, {final_predictions['up_prob_1d'].max():.3f}]")
        logger.info(f"{TrainingLogMessages.PREDICTION_STATISTICS.value} - 20-day return range: [{final_predictions['return_20d'].min():.4f}, {final_predictions['return_20d'].max():.4f}]")
        logger.info(f"{TrainingLogMessages.PREDICTION_STATISTICS.value} - 20-day probability range: [{final_predictions['up_prob_20d'].min():.3f}, {final_predictions['up_prob_20d'].max():.3f}]")

        logger.info(TrainingLogMessages.TRAINING_SCRIPT_SUCCESS.value)

    except Exception as e:
        logger.error(f"{TrainingLogMessages.TRAINING_SCRIPT_ERROR.value} - {e}")
        import traceback
        logger.error(f"{TrainingLogMessages.TRAINING_SCRIPT_ERROR.value} - Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
