import os

class Config:
    # Директории
    DATA_DIR = './data/'
    RAW_DATA_DIR = './data/raw/'
    PROCESSED_DATA_DIR = './data/processed/'
    MODELS_DIR = './models/'
    PREDICTIONS_DIR = './predictions/'
    ARTIFACTS_DIR = './artifacts/'
    REPORTS_DIR = './reports/'

    # Параметры модели
    SEED = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 200
    MAX_DEPTH = 15
    RSI_WINDOW = 14
    VOLATILITY_WINDOWS = [5, 10, 20]
    SMA_WINDOWS = [5, 10, 20, 50]
    FORECAST_HORIZONS = [1, 20]

    # Ключевые слова для тикеров
    TICKER_KEYWORDS = {
        'AFLT': ['Aeroflot', 'AFLT', 'аэрофлот', 'аэрофлота'],
        'ALRS': ['ALROSA', 'ALRS', 'алроса'],
        'CHMF': ['Severstal', 'CHMF', 'северсталь'],
        'GAZP': ['Gazprom', 'GAZP', 'газпром'],
        'GMKN': ['Norilsk Nickel', 'GMKN', 'норильский никель', 'норникель'],
        'LKOH': ['Lukoil', 'LKOH', 'лукойл'],
        'MAGN': ['Magnitogorsk', 'MAGN', 'магнитогорск'],
        'MGNT': ['Magnit', 'MGNT', 'магнит'],
        'MOEX': ['Moscow Exchange', 'MOEX', 'мосбиржа'],
        'MTSS': ['MTS', 'MTSS', 'мтс'],
        'NVTK': ['Novatek', 'NVTK', 'новатэк'],
        'PHOR': ['PhosAgro', 'PHOR', 'фосагро'],
        'PLZL': ['Polyus', 'PLZL', 'полюс'],
        'ROSN': ['Rosneft', 'ROSN', 'росснефть'],
        'RUAL': ['Rusal', 'RUAL', 'русал'],
        'SBER': ['Sberbank', 'SBER', 'сбербанк'],
        'SIBN': ['Sibur', 'SIBN', 'сибур'],
        'TATN': ['Tatneft', 'TATN', 'T', 'татнефть'],  # Исправлено с 'T' на 'TATN'
        'VTBR': ['VTB', 'VTBR', 'втб']
    }

    # Для обратной совместимости (если нужно)
    @property
    def data_dir(self):
        return self.DATA_DIR

    @property
    def raw_data_dir(self):
        return self.RAW_DATA_DIR

    @property
    def processed_data_dir(self):
        return self.PROCESSED_DATA_DIR

    @property
    def models_dir(self):
        return self.MODELS_DIR

    @property
    def predictions_dir(self):
        return self.PREDICTIONS_DIR

    @property
    def artifacts_dir(self):
        return self.ARTIFACTS_DIR

    @property
    def reports_dir(self):
        return self.REPORTS_DIR

    @property
    def seed(self):
        return self.SEED

    @property
    def test_size(self):
        return self.TEST_SIZE

    @property
    def n_estimators(self):
        return self.N_ESTIMATORS

    @property
    def max_depth(self):
        return self.MAX_DEPTH

    @property
    def rsi_window(self):
        return self.RSI_WINDOW

    @property
    def volatility_windows(self):
        return self.VOLATILITY_WINDOWS

    @property
    def sma_windows(self):
        return self.SMA_WINDOWS

    @property
    def forecast_horizons(self):
        return self.FORECAST_HORIZONS

    @property
    def ticker_keywords(self):
        return self.TICKER_KEYWORDS
