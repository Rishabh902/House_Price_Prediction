from pathlib import Path
import joblib
import pandas as pd
import logging

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "linear_regression.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
COLUMNS_PATH = BASE_DIR / "model" / "columns.pkl"
LOG_PATH = BASE_DIR / "model" / "app.log"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
columns = joblib.load(COLUMNS_PATH)

def predict_house_price(data: dict):
    logging.info(f"Received input: {data}")

    df = pd.DataFrame([data])

    #same column order as training
    df = df.reindex(columns=columns, fill_value=0)

    logging.info(f"Before scaling:\n{df}")

    #IMPORTANT: scale before prediction
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]

    logging.info(f"Raw prediction: {prediction}")

    final_prediction = max(0, round(float(prediction), 2))

    logging.info(f"Final prediction: {final_prediction}")

    return final_prediction

