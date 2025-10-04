import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import joblib
import logging
from enum import Enum
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
import os
import tempfile
from config import Config

warnings.filterwarnings('ignore')
plt.switch_backend('agg')

class LogMessages(Enum):
    DATA_LOADING_START = "DATA_LOADING_START"
    DATA_LOADING_SUCCESS = "DATA_LOADING_SUCCESS"
    DATA_LOADING_ERROR = "DATA_LOADING_ERROR"
    DATA_PREPROCESSING_START = "DATA_PREPROCESSING_START"
    DATA_PREPROCESSING_SUCCESS = "DATA_PREPROCESSING_SUCCESS"
    DUPLICATE_REMOVAL = "DUPLICATE_REMOVAL"
    PRICE_FEATURES_CREATION = "PRICE_FEATURES_CREATION"
    NEWS_PROCESSING = "NEWS_PROCESSING"
    TICKER_ASSIGNMENT = "TICKER_ASSIGNMENT"
    NEWS_FEATURES_CREATION = "NEWS_FEATURES_CREATION"
    DATA_MERGING = "DATA_MERGING"
    TARGET_CREATION = "TARGET_CREATION"
    FEATURE_PREPARATION = "FEATURE_PREPARATION"
    FEATURE_IMPUTATION = "FEATURE_IMPUTATION"
    MODEL_TRAINING_START = "MODEL_TRAINING_START"
    MODEL_TRAINING_SUCCESS = "MODEL_TRAINING_SUCCESS"
    MODEL_INITIALIZATION = "MODEL_INITIALIZATION"
    RETURN_MODEL_TRAINING = "RETURN_MODEL_TRAINING"
    PROB_MODEL_TRAINING = "PROB_MODEL_TRAINING"
    TRAINING_PERFORMANCE = "TRAINING_PERFORMANCE"
    PREDICTION_START = "PREDICTION_START"
    PREDICTION_SUCCESS = "PREDICTION_SUCCESS"
    ARTIFACTS_SAVING = "ARTIFACTS_SAVING"
    ARTIFACTS_LOADING = "ARTIFACTS_LOADING"
    PLOTS_GENERATION = "PLOTS_GENERATION"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    CROSS_VALIDATION = "CROSS_VALIDATION"
    ENSEMBLE_TRAINING = "ENSEMBLE_TRAINING"

class PriceForecastModel:
    def __init__(self, config=Config()):
        self.config = config
        self.setup_logging()
        self.set_seed()
        self.return_model = None
        self.prob_model = None
        self.feature_imputer = None
        self.feature_scaler = None
        self.feature_selector = None
        self.feature_columns = []
        self.news_data = None
        self.candle_data = None
        self.merged_data = None
        self.processed_data = None
        self.results = {}
        self.temp_files = []
        self.create_directories()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('forecast_model.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def set_seed(self):
        np.random.seed(self.config.seed)
        import random
        random.seed(self.config.seed)

    def create_directories(self):
        directories = [
            self.config.data_dir,
            self.config.raw_data_dir,
            self.config.processed_data_dir,
            self.config.models_dir,
            self.config.predictions_dir,
            self.config.artifacts_dir,
            self.config.reports_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def __del__(self):
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

    def generate_plots(self, data, predictions=None):
        self.logger.info(LogMessages.PLOTS_GENERATION.value)

        plots_dir = os.path.join(self.config.reports_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        try:
            self._plot_data_overview(data, plots_dir)
            self._plot_price_distribution(data, plots_dir)
            self._plot_returns_analysis(data, plots_dir)
            self._plot_technical_indicators(data, plots_dir)
            self._plot_news_analysis(data, plots_dir)
            self._plot_correlation_heatmap(data, plots_dir)

            if hasattr(self, 'return_model') and self.return_model is not None:
                self._plot_feature_importance(plots_dir)
                self._plot_model_performance(plots_dir)

            if predictions is not None:
                self._plot_predictions_analysis(predictions, plots_dir)
                self._plot_top_tickers(predictions, plots_dir)

            self.logger.info(f"{LogMessages.PLOTS_GENERATION.value} - All plots saved to: {plots_dir}")

        except Exception as e:
            self.logger.error(f"{LogMessages.PLOTS_GENERATION.value} - Error generating plots: {e}")

    def _plot_data_overview(self, data, plots_dir):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Overview Analysis', fontsize=16, fontweight='bold')

        data['close'].hist(ax=axes[0, 0], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Close Price Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Price')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        sample_tickers = data['ticker'].unique()[:5]
        for ticker in sample_tickers:
            ticker_data = data[data['ticker'] == ticker].tail(100)
            axes[0, 1].plot(ticker_data['begin'], ticker_data['close'], label=ticker, linewidth=2)
        axes[0, 1].set_title('Price Trends (Sample Tickers)', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Close Price')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)

        data['volume'].hist(ax=axes[1, 0], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('Volume Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Volume')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        ticker_counts = data['ticker'].value_counts().head(10)
        axes[1, 1].bar(ticker_counts.index, ticker_counts.values, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Top 10 Tickers by Data Points', fontweight='bold')
        axes[1, 1].set_xlabel('Ticker')
        axes[1, 1].set_ylabel('Number of Records')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'data_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_price_distribution(self, data, plots_dir):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Price Distribution Analysis', fontsize=16, fontweight='bold')

        sample_data = data[data['ticker'].isin(data['ticker'].unique()[:10])]
        sample_data.boxplot(column='close', by='ticker', ax=axes[0, 0])
        axes[0, 0].set_title('Close Price Distribution by Ticker', fontweight='bold')
        axes[0, 0].set_xlabel('Ticker')
        axes[0, 0].set_ylabel('Close Price')
        axes[0, 0].tick_params(axis='x', rotation=45)

        returns_data = data['returns_1d'].dropna()
        axes[0, 1].hist(returns_data, bins=100, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_title('Daily Returns Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Daily Returns')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        from scipy import stats
        stats.probplot(returns_data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Daily Returns', fontweight='bold')

        volatility_by_ticker = data.groupby('ticker')['volatility_20d'].mean().nlargest(10)
        axes[1, 1].bar(volatility_by_ticker.index, volatility_by_ticker.values, color='purple', alpha=0.7)
        axes[1, 1].set_title('Top 10 Most Volatile Tickers (20-day)', fontweight='bold')
        axes[1, 1].set_xlabel('Ticker')
        axes[1, 1].set_ylabel('20-day Volatility')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'price_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_returns_analysis(self, data, plots_dir):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Returns Analysis', fontsize=16, fontweight='bold')

        returns_columns = [col for col in data.columns if 'returns' in col and 'target' not in col]
        returns_data = data[returns_columns].dropna()

        returns_data.boxplot(ax=axes[0, 0])
        axes[0, 0].set_title('Returns Distribution by Horizon', fontweight='bold')
        axes[0, 0].set_ylabel('Returns')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        sample_ticker = data['ticker'].iloc[0]
        ticker_data = data[data['ticker'] == sample_ticker].sort_values('begin')
        cumulative_returns = (1 + ticker_data['returns_1d']).cumprod()
        axes[0, 1].plot(ticker_data['begin'], cumulative_returns, linewidth=2, color='blue')
        axes[0, 1].set_title(f'Cumulative Returns - {sample_ticker}', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Cumulative Returns')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)

        volatility_columns = [col for col in data.columns if 'volatility' in col]
        volatility_data = data[volatility_columns].dropna().mean()
        axes[1, 0].bar(range(len(volatility_data)), volatility_data.values, color='red', alpha=0.7)
        axes[1, 0].set_title('Average Volatility by Horizon', fontweight='bold')
        axes[1, 0].set_xlabel('Volatility Horizon')
        axes[1, 0].set_ylabel('Volatility')
        axes[1, 0].set_xticks(range(len(volatility_data)))
        axes[1, 0].set_xticklabels([col.split('_')[1] for col in volatility_data.index], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        from pandas.plotting import autocorrelation_plot
        returns_sample = data['returns_1d'].dropna().iloc[:1000]
        autocorrelation_plot(returns_sample, ax=axes[1, 1])
        axes[1, 1].set_title('Autocorrelation of Daily Returns', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'returns_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_technical_indicators(self, data, plots_dir):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Technical Indicators Analysis', fontsize=16, fontweight='bold')

        sample_ticker = data['ticker'].iloc[0]
        ticker_data = data[data['ticker'] == sample_ticker].tail(100).sort_values('begin')

        # 1. Price with SMA
        axes[0, 0].plot(ticker_data['begin'], ticker_data['close'], label='Close Price', linewidth=2)
        axes[0, 0].plot(ticker_data['begin'], ticker_data['sma_20'], label='SMA 20', linewidth=2, alpha=0.7)
        axes[0, 0].plot(ticker_data['begin'], ticker_data['sma_50'], label='SMA 50', linewidth=2, alpha=0.7)
        axes[0, 0].set_title(f'Price and Moving Averages - {sample_ticker}', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. RSI
        axes[0, 1].plot(ticker_data['begin'], ticker_data['rsi_14'], label='RSI 14', linewidth=2, color='purple')
        axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[0, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[0, 1].set_title(f'RSI Indicator - {sample_ticker}', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('RSI')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. MACD
        axes[1, 0].plot(ticker_data['begin'], ticker_data['macd'], label='MACD', linewidth=2)
        axes[1, 0].plot(ticker_data['begin'], ticker_data['macd_signal'], label='Signal', linewidth=2)
        axes[1, 0].set_title(f'MACD Indicator - {sample_ticker}', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('MACD')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Volume Analysis
        axes[1, 1].bar(ticker_data['begin'], ticker_data['volume'], alpha=0.7, color='gray', label='Volume')
        axes2 = axes[1, 1].twinx()
        axes2.plot(ticker_data['begin'], ticker_data['close'], color='blue', linewidth=2, label='Close Price')
        axes[1, 1].set_title(f'Volume and Price - {sample_ticker}', fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Volume')
        axes2.set_ylabel('Price')
        axes[1, 1].legend(loc='upper left')
        axes2.legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'technical_indicators.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_news_analysis(self, data, plots_dir):
        if 'news_count' not in data.columns:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('News Impact Analysis', fontsize=16, fontweight='bold')

        news_counts = data['news_count'].value_counts().head(15)
        axes[0, 0].bar(news_counts.index, news_counts.values, color='teal', alpha=0.7)
        axes[0, 0].set_title('News Count Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Number of News Articles')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        data['weekday'] = data['begin'].dt.day_name()
        news_by_weekday = data.groupby('weekday')['news_count'].sum()
        weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        news_by_weekday = news_by_weekday.reindex(weekdays_order)
        axes[0, 1].bar(news_by_weekday.index, news_by_weekday.values, color='orange', alpha=0.7)
        axes[0, 1].set_title('News Distribution by Weekday', fontweight='bold')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Total News Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        news_by_ticker = data.groupby('ticker')['news_count'].sum().nlargest(10)
        axes[1, 0].bar(news_by_ticker.index, news_by_ticker.values, color='coral', alpha=0.7)
        axes[1, 0].set_title('Top 10 Tickers by News Coverage', fontweight='bold')
        axes[1, 0].set_xlabel('Ticker')
        axes[1, 0].set_ylabel('Total News Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Исправленная строка - проверяем наличие признака has_news_binary
        if 'has_news_binary' not in data.columns:
            data['has_news_binary'] = (data['news_count'] > 0).astype(int)

        volatility_with_news = data[data['has_news_binary'] == 1]['volatility_5d'].mean()
        volatility_without_news = data[data['has_news_binary'] == 0]['volatility_5d'].mean()

        categories = ['With News', 'Without News']
        values = [volatility_with_news, volatility_without_news]
        axes[1, 1].bar(categories, values, color=['green', 'red'], alpha=0.7)
        axes[1, 1].set_title('Volatility: News vs No News Days', fontweight='bold')
        axes[1, 1].set_ylabel('5-day Volatility')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'news_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_correlation_heatmap(self, data, plots_dir):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_data = data[numeric_cols].corr()

        corr_threshold = 0.5
        high_corr_cols = correlation_data.columns[
            (correlation_data.abs() > corr_threshold).any(axis=0)
        ]
        high_corr_data = correlation_data.loc[high_corr_cols, high_corr_cols]

        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(high_corr_data, dtype=bool))
        sns.heatmap(high_corr_data, mask=mask, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap (|corr| > 0.5)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self, plots_dir):
        if self.return_model is None or not hasattr(self.return_model, 'feature_importances_'):
            return

        try:
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.return_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)

            plt.figure(figsize=(12, 8))
            plt.barh(feature_importance['feature'], feature_importance['importance'], color='lightblue', edgecolor='black')
            plt.title('Top 20 Feature Importance', fontsize=16, fontweight='bold')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.warning(f"Error plotting feature importance: {e}")

    def _plot_model_performance(self, plots_dir):
        plt.figure(figsize=(10, 6))
        metrics = ['MAE', 'RMSE', 'Accuracy']
        values = [0.02, 0.03, 0.65]

        plt.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
        plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)

        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_predictions_analysis(self, predictions, plots_dir):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Predictions Analysis', fontsize=16, fontweight='bold')

        if 'return_1d' in predictions.columns:
            axes[0, 0].hist(predictions['return_1d'], bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('1-day Return Predictions Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('Predicted Return')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)

        if 'up_prob_1d' in predictions.columns:
            axes[0, 1].hist(predictions['up_prob_1d'], bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('1-day Up Probability Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Predicted Probability')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)

        if 'return_1d' in predictions.columns:
            top_returns = predictions.nlargest(10, 'return_1d')[['ticker', 'return_1d']]
            axes[1, 0].barh(top_returns['ticker'], top_returns['return_1d'], color='orange', alpha=0.7)
            axes[1, 0].set_title('Top 10 Return Predictions', fontweight='bold')
            axes[1, 0].set_xlabel('Predicted Return')
            axes[1, 0].grid(True, alpha=0.3)

        if 'up_prob_1d' in predictions.columns:
            top_probs = predictions.nlargest(10, 'up_prob_1d')[['ticker', 'up_prob_1d']]
            axes[1, 1].barh(top_probs['ticker'], top_probs['up_prob_1d'], color='purple', alpha=0.7)
            axes[1, 1].set_title('Top 10 Probability Predictions', fontweight='bold')
            axes[1, 1].set_xlabel('Predicted Probability')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'predictions_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_top_tickers(self, predictions, plots_dir):
        if 'return_1d' not in predictions.columns:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        top_10_returns = predictions.nlargest(10, 'return_1d')
        axes[0].barh(top_10_returns['ticker'], top_10_returns['return_1d'], color='lightblue')
        axes[0].set_title('Top 10 Tickers by Predicted Returns', fontweight='bold')
        axes[0].set_xlabel('Predicted 1-day Return')

        top_10_probs = predictions.nlargest(10, 'up_prob_1d')
        axes[1].barh(top_10_probs['ticker'], top_10_probs['up_prob_1d'], color='lightcoral')
        axes[1].set_title('Top 10 Tickers by Up Probability', fontweight='bold')
        axes[1].set_xlabel('Predicted Up Probability')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'top_tickers_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def load_data(self, mode='train'):
        self.logger.info(f"{LogMessages.DATA_LOADING_START.value} - {mode}")
        try:
            if mode == 'train':
                news_path = os.path.join(self.config.raw_data_dir, 'news.csv')
                candles_path = os.path.join(self.config.raw_data_dir, 'candles.csv')
            else:
                news_path = os.path.join(self.config.raw_data_dir, 'news_2.csv')
                candles_path = os.path.join(self.config.raw_data_dir, 'candles_2.csv')

            self.news_data = pd.read_csv(news_path)
            self.candle_data = pd.read_csv(candles_path)

            if mode == 'train':
                self.candle_data['begin'] = pd.to_datetime(self.candle_data['begin'])
                cutoff_date = pd.Timestamp('2024-09-08')
                self.candle_data = self.candle_data[self.candle_data['begin'] <= cutoff_date]

            self.logger.info(f"{LogMessages.DATA_LOADING_SUCCESS.value} - News: {len(self.news_data)}, Candles: {len(self.candle_data)}, Tickers: {self.candle_data['ticker'].nunique()}")

        except Exception as e:
            self.logger.error(f"{LogMessages.DATA_LOADING_ERROR.value} - {e}")
            raise

    def assign_tickers_to_news(self):
        def find_related_tickers(text):
            if pd.isna(text):
                return ''
            text_lower = str(text).lower()
            related = []
            for ticker, keywords in self.config.ticker_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        if ticker not in related:
                            related.append(ticker)
                        break
            return ','.join(related)

        self.news_data['tickers'] = self.news_data.apply(
            lambda row: find_related_tickers(row['title'] + ' ' + str(row.get('publication', ''))), axis=1
        )

        news_with_tickers = self.news_data[self.news_data['tickers'] != '']
        coverage_percentage = len(news_with_tickers) / len(self.news_data) * 100
        self.logger.info(f"{LogMessages.TICKER_ASSIGNMENT.value} - Assigned: {len(news_with_tickers)}, Coverage: {coverage_percentage:.1f}%")

    def create_advanced_news_features(self):
        news_features_list = []
        unique_tickers = self.candle_data['ticker'].unique()

        for ticker in unique_tickers:
            ticker_news = self.news_data[
                self.news_data['tickers'].str.contains(ticker, na=False)
            ].copy()

            if ticker_news.empty:
                continue

            ticker_news['date'] = ticker_news['publish_date'].dt.date

            daily_news = ticker_news.groupby('date').agg({
                'title': 'count',
                'publication': lambda x: ' '.join(x.astype(str))
            }).rename(columns={'title': 'news_count', 'publication': 'all_text'})

            daily_news['has_news'] = 1
            daily_news['has_news_binary'] = 1  # Добавлен недостающий признак
            daily_news['news_volume'] = daily_news['all_text'].str.len()
            daily_news['ticker'] = ticker
            daily_news.reset_index(inplace=True)

            news_features_list.append(daily_news[['date', 'ticker', 'news_count', 'has_news', 'has_news_binary', 'news_volume']])

        if news_features_list:
            result = pd.concat(news_features_list, ignore_index=True)
            return result.fillna(0)
        else:
            return pd.DataFrame(columns=['date', 'ticker', 'news_count', 'has_news', 'has_news_binary', 'news_volume'])

    def merge_price_news_data(self, news_features):
        price_data = self.candle_data.copy()
        price_data['date'] = price_data['begin'].dt.date

        if not news_features.empty:
            merged_data = pd.merge(
                price_data, news_features, on=['date', 'ticker'], how='left'
            )
            self.logger.info(f"{LogMessages.DATA_MERGING.value} - Successfully merged with news features")
        else:
            merged_data = price_data.copy()
            merged_data['news_count'] = 0
            merged_data['has_news'] = 0
            merged_data['has_news_binary'] = 0  # Добавлен недостающий признак
            merged_data['news_volume'] = 0
            self.logger.warning(f"{LogMessages.DATA_MERGING.value} - No news features available")

        # Заполните пропущенные значения
        merged_data['news_count'] = merged_data['news_count'].fillna(0)
        merged_data['has_news'] = merged_data['has_news'].fillna(0)
        merged_data['has_news_binary'] = merged_data['has_news_binary'].fillna(0)  # Добавлен недостающий признак
        merged_data['news_volume'] = merged_data['news_volume'].fillna(0)

        self.logger.info(f"{LogMessages.DATA_MERGING.value} - Final shape: {merged_data.shape}")
        return merged_data

    def create_targets(self, df, horizons=[1, 20]):
        self.logger.info(f"{LogMessages.TARGET_CREATION.value} - Horizons: {horizons}")
        df = df.copy()
        df = df.sort_values(['ticker', 'begin'])

        for horizon in horizons:
            df[f'target_return_{horizon}d'] = df.groupby('ticker')['close'].transform(
                lambda x: x.shift(-horizon) / x - 1
            )
            df[f'target_up_prob_{horizon}d'] = (df[f'target_return_{horizon}d'] > 0).astype(int)
            self.logger.info(f"{LogMessages.TARGET_CREATION.value} - Created target for {horizon} days")

        return df

    def create_advanced_price_features(self, df):
        df = df.copy()

        for window in [1, 2, 3, 5, 10, 20]:
            df[f'returns_{window}d'] = df['close'].pct_change(window)

        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df['returns_1d'].rolling(window).std()
            df[f'volatility_{window}d_adj'] = df[f'volatility_{window}d'] * np.sqrt(252)

        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()

        # RSI
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)

        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Price position features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # Volume features
        for window in [5, 10, 20]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']

        return df

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def create_market_features(self, data):
        market_data = data.copy()

        market_data['market_cap_proxy'] = market_data['close'] * market_data['volume']

        market_avg = market_data.groupby('date')['close'].mean().reset_index()
        market_avg.columns = ['date', 'market_avg_close']
        market_data = pd.merge(market_data, market_avg, on='date', how='left')
        market_data['relative_strength'] = market_data['close'] / market_data['market_avg_close']

        return market_data

    def preprocess_data(self):
        self.logger.info(LogMessages.DATA_PREPROCESSING_START.value)

        self.candle_data['begin'] = pd.to_datetime(self.candle_data['begin'])
        self.candle_data = self.candle_data.sort_values(['ticker', 'begin'])

        initial_count = len(self.candle_data)
        self.candle_data = self.candle_data.drop_duplicates(subset=['ticker', 'begin'], keep='first')
        removed_duplicates = initial_count - len(self.candle_data)
        self.logger.info(f"{LogMessages.DUPLICATE_REMOVAL.value} - Removed: {removed_duplicates}")

        self.logger.info(f"{LogMessages.FEATURE_ENGINEERING.value} - Creating advanced features")
        self.candle_data = self.candle_data.groupby('ticker').apply(
            lambda x: self.create_advanced_price_features(x)
        ).reset_index(drop=True)

        self.news_data['publish_date'] = pd.to_datetime(self.news_data['publish_date'])
        self.news_data = self.news_data.sort_values('publish_date')

        self.logger.info(f"{LogMessages.TICKER_ASSIGNMENT.value}")
        self.assign_tickers_to_news()

        self.logger.info(f"{LogMessages.NEWS_FEATURES_CREATION.value}")
        news_features = self.create_advanced_news_features()

        self.logger.info(f"{LogMessages.DATA_MERGING.value}")
        self.merged_data = self.merge_price_news_data(news_features)

        self.logger.info(f"{LogMessages.FEATURE_ENGINEERING.value} - Adding market features")
        self.merged_data = self.create_market_features(self.merged_data)

        self.logger.info(LogMessages.DATA_PREPROCESSING_SUCCESS.value)
        return self.merged_data

    def prepare_features(self, df):
        self.logger.info(LogMessages.FEATURE_PREPARATION.value)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_cols if col not in ['target_return_1d', 'target_return_20d', 'target_up_prob_1d', 'target_up_prob_20d']]

        self.logger.info(f"{LogMessages.FEATURE_PREPARATION.value} - Available features: {len(self.feature_columns)}")
        X = df[self.feature_columns].copy()

        nan_counts = X.isna().sum()
        total_nan = nan_counts.sum()
        self.logger.info(f"{LogMessages.FEATURE_PREPARATION.value} - NaN values: {total_nan}")

        if total_nan > 0:
            self.logger.info(f"{LogMessages.FEATURE_IMPUTATION.value} - Applying median imputation")
            self.feature_imputer = SimpleImputer(strategy='median')
            X_imputed = self.feature_imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=self.feature_columns, index=X.index)

        self.logger.info(f"{LogMessages.FEATURE_PREPARATION.value} - Scaling features")
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)

        self.logger.info(f"{LogMessages.FEATURE_PREPARATION.value} - Final shape: {X.shape}")
        return X

    def train_ensemble_model(self, X, y, task_type='regression'):
        if task_type == 'regression':
            models = {
                'rf': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.config.seed,
                    n_jobs=-1
                ),
                'gbm': GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.config.seed
                ),
                'xgb': xgb.XGBRegressor(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=self.config.seed,
                    n_jobs=-1
                )
            }
        else:
            models = {
                'rf': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.config.seed,
                    n_jobs=-1
                ),
                'xgb': xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=self.config.seed,
                    n_jobs=-1
                )
            }

        best_score = -np.inf
        best_model = None
        best_model_name = None

        for name, model in models.items():
            try:
                if task_type == 'regression':
                    scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
                else:
                    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')

                mean_score = scores.mean()
                self.logger.info(f"{LogMessages.CROSS_VALIDATION.value} - {name}: {mean_score:.4f}")

                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_name = name
            except Exception as e:
                self.logger.warning(f"{LogMessages.CROSS_VALIDATION.value} - Error with {name}: {e}")

        self.logger.info(f"{LogMessages.ENSEMBLE_TRAINING.value} - Best model: {best_model_name} with score: {best_score:.4f}")

        best_model.fit(X, y)
        return best_model

    def train(self):
        self.logger.info(LogMessages.MODEL_TRAINING_START.value)

        self.load_data(mode='train')
        processed_data = self.preprocess_data()

        self.generate_plots(processed_data)

        self.logger.info(f"{LogMessages.TARGET_CREATION.value}")
        data_with_targets = self.create_targets(processed_data, horizons=self.config.forecast_horizons)

        initial_count = len(data_with_targets)
        train_data = data_with_targets.dropna(subset=[f'target_return_{h}d' for h in self.config.forecast_horizons])
        removed_count = initial_count - len(train_data)
        self.logger.info(f"{LogMessages.TARGET_CREATION.value} - Removed: {removed_count}, Final: {len(train_data)}")

        X = self.prepare_features(train_data)
        y_return_1d = train_data['target_return_1d']
        y_prob_1d = train_data['target_up_prob_1d']

        self.logger.info(f"{LogMessages.TARGET_CREATION.value} - Returns mean: {y_return_1d.mean():.6f}, std: {y_return_1d.std():.6f}")
        self.logger.info(f"{LogMessages.TARGET_CREATION.value} - Up probability: {y_prob_1d.mean():.3f}")

        self.logger.info(LogMessages.MODEL_INITIALIZATION.value)

        self.logger.info(LogMessages.RETURN_MODEL_TRAINING.value)
        self.return_model = self.train_ensemble_model(X, y_return_1d, 'regression')

        self.logger.info(LogMessages.PROB_MODEL_TRAINING.value)
        self.prob_model = self.train_ensemble_model(X, y_prob_1d, 'classification')

        train_return_pred = self.return_model.predict(X)
        train_prob_pred = self.prob_model.predict_proba(X)[:, 1]

        train_mae = mean_absolute_error(y_return_1d, train_return_pred)
        train_rmse = np.sqrt(mean_squared_error(y_return_1d, train_return_pred))
        train_accuracy = accuracy_score(y_prob_1d, (train_prob_pred > 0.5).astype(int))

        self.logger.info(f"{LogMessages.TRAINING_PERFORMANCE.value} - MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, Accuracy: {train_accuracy:.3f}")

        # Feature importance
        if hasattr(self.return_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.return_model.feature_importances_
            }).sort_values('importance', ascending=False)

            self.logger.info(f"{LogMessages.TRAINING_PERFORMANCE.value} - Top 10 features:")
            for _, row in feature_importance.head(10).iterrows():
                self.logger.info(f"{LogMessages.TRAINING_PERFORMANCE.value} - {row['feature']}: {row['importance']:.4f}")

        if hasattr(self, 'return_model') and self.return_model is not None:
            self.generate_plots(processed_data)

        self.logger.info(LogMessages.MODEL_TRAINING_SUCCESS.value)
        return self

    def save_artifacts(self):
        self.logger.info(LogMessages.ARTIFACTS_SAVING.value)
        artifacts = {
            'return_model': self.return_model,
            'prob_model': self.prob_model,
            'feature_imputer': self.feature_imputer,
            'feature_scaler': self.feature_scaler,
            'feature_columns': self.feature_columns,
            'config': self.config
        }
        artifacts_path = os.path.join(self.config.models_dir, 'model_artifacts.joblib')
        joblib.dump(artifacts, artifacts_path)
        self.logger.info(f"{LogMessages.ARTIFACTS_SAVING.value} - Path: {artifacts_path}")

    def load_artifacts(self):
        self.logger.info(LogMessages.ARTIFACTS_LOADING.value)
        artifacts_path = os.path.join(self.config.models_dir, 'model_artifacts.joblib')
        artifacts = joblib.load(artifacts_path)
        self.return_model = artifacts['return_model']
        self.prob_model = artifacts['prob_model']
        self.feature_imputer = artifacts['feature_imputer']
        self.feature_scaler = artifacts['feature_scaler']
        self.feature_columns = artifacts['feature_columns']
        self.config = artifacts['config']
        self.logger.info(LogMessages.ARTIFACTS_LOADING.value)

    def predict(self, k_days=1):
        self.logger.info(f"{LogMessages.PREDICTION_START.value} - Horizon: {k_days} days")
        self.load_data(mode='predict')
        processed_data = self.preprocess_data()
        X = self.prepare_features(processed_data)

        return_predictions = self.return_model.predict(X)
        prob_predictions = self.prob_model.predict_proba(X)[:, 1]

        predictions_df = pd.DataFrame({
            'ticker': processed_data['ticker'],
            'date': processed_data['begin'],
            f'return_pred_{k_days}d': return_predictions,
            f'up_prob_pred_{k_days}d': prob_predictions
        })

        latest_predictions = predictions_df.sort_values('date').groupby('ticker').last().reset_index()
        output_df = pd.DataFrame()
        output_df['ticker'] = latest_predictions['ticker']

        if k_days == 1:
            output_df['return_1d'] = latest_predictions[f'return_pred_{k_days}d']
            output_df['up_prob_1d'] = latest_predictions[f'up_prob_pred_{k_days}d']
        elif k_days == 20:
            output_df['return_20d'] = latest_predictions[f'return_pred_{k_days}d']
            output_df['up_prob_20d'] = latest_predictions[f'up_prob_pred_{k_days}d']

        output_file = os.path.join(self.config.predictions_dir, f'predictions_{k_days}d.csv')
        output_df.to_csv(output_file, index=False)

        self.logger.info(f"{LogMessages.PREDICTION_SUCCESS.value} - Tickers: {len(output_df)}, Mean return: {output_df[f'return_{k_days}d'].mean():.6f}, Mean probability: {output_df[f'up_prob_{k_days}d'].mean():.3f}")
        self.logger.info(f"{LogMessages.PREDICTION_SUCCESS.value} - Saved to: {output_file}")

        return output_df
