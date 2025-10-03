#!/bin/bash
# run.sh - скрипт для запуска пайплайна

echo "Запуск Forex News Predictor"

# Режим обучения
echo "Обучение модели..."
python forex_predictor.py --mode train \
    --candles data/task_1_candles.csv \
    --news data/task_1_news.csv \
    --artifacts_dir artifacts

# Режим предсказания
echo "Создание предсказаний..."
python forex_predictor.py --mode predict \
    --candles data/test_candles.csv \
    --news data/test_news.csv \
    --output submission.csv \
    --artifacts_dir artifacts

echo "✅ Готово! Проверьте файл submission.csv"
