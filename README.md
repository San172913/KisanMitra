# KisanMitra — Simple Version

A beginner-friendly crop recommendation ML project using four models: KNN, Decision Tree, SVM and Random Forest.

## Run on Windows
1. Put `Crop_recommendation.csv` in the `data` folder.
2. Double-click `setup_windows.bat` to install packages and train the models.
3. Double-click `run_app.bat` to open the Streamlit app.

If the dataset download does not work, download the Crop Recommendation dataset manually and save it as `data/Crop_recommendation.csv`.

## Run commands manually
```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training script prints basic dataset checks, cleans missing/duplicate rows, makes a bar chart, splits data 80:20, tunes four models using 5-fold cross-validation on the training set, and compares test metrics. It saves the best model and results in `models` and `outputs`.

## Inputs
N, P, K, temperature, humidity, pH and rainfall. The model recommendation is educational and should not replace advice from local agricultural experts.
