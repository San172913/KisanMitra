from pathlib import Path
from io import BytesIO
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
URLS = [
    "https://raw.githubusercontent.com/niteshwayakar/Crop-Recommendation/main/Crop_recommendation.csv",
    "https://raw.githubusercontent.com/ravikant-diwakar/AgriSens/master/Datasets/Crop_recommendation.csv",
    "https://huggingface.co/spaces/Vivswaan/crop_recommendation/resolve/main/Crop_recommendation.csv",
]
REQUIRED = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"}

DATA_PATH.parent.mkdir(exist_ok=True)
if DATA_PATH.exists():
    print("Dataset already exists:", DATA_PATH)
    raise SystemExit(0)

for url in URLS:
    try:
        print("Trying:", url)
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        df = pd.read_csv(BytesIO(response.content))
        df.columns = [str(c).strip().lower() for c in df.columns]
        df.rename(columns={
            "n": "N", "p": "P", "k": "K", "nitrogen": "N",
            "phosphorus": "P", "potassium": "K", "ph_value": "ph",
            "crop": "label"
        }, inplace=True)
        if not REQUIRED.issubset(set(df.columns)) or len(df) < 1000:
            raise ValueError("Downloaded file does not look like the Crop Recommendation dataset.")
        df.to_csv(DATA_PATH, index=False)
        print("Dataset saved:", DATA_PATH)
        print("Rows and columns:", df.shape)
        break
    except Exception as error:
        print("Download failed:", error)
else:
    print("\nCould not download the dataset.")
    print("Open README.md and save the CSV manually to:")
    print(DATA_PATH)
    raise SystemExit(1)
