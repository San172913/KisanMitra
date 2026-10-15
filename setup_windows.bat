@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if not exist "data\Crop_recommendation.csv" python download_dataset.py
if not exist "data\Crop_recommendation.csv" (
  echo Dataset not found. Download Crop_recommendation.csv and place it in the data folder.
  pause
  exit /b 1
)
python train_model.py
pause
