from datetime import datetime
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

FLIGHT_MODEL_PATH = MODELS_DIR / "flight_price_model.pkl"
GENDER_MODEL_PATH = MODELS_DIR / "gender_classifier.pkl"
HOTEL_RECOMMENDER_PATH = MODELS_DIR / "hotel_recommender.pkl"

FLIGHT_REQUIRED_FIELDS = ["from", "to", "flightType", "time", "distance", "agency"]
FLIGHT_FEATURE_COLUMNS = [
    "from",
    "to",
    "flightType",
    "time",
    "distance",
    "agency",
    "flight_month",
    "flight_day",
    "flight_dayofweek",
    "is_weekend",
]

_ARTIFACT_CACHE = {}

app = Flask(__name__)


def _load_artifact(name, path):
    if name not in _ARTIFACT_CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} was not found. Run `python train_models.py` first."
            )
        _ARTIFACT_CACHE[name] = joblib.load(path)
    return _ARTIFACT_CACHE[name]


def _json_error(message, status_code=400, **extra):
    body = {"error": message}
    body.update(extra)
    return jsonify(body), status_code


def _records_from_payload(payload):
    if isinstance(payload, dict):
        return [payload], True
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload, False
    raise ValueError("Request body must be a JSON object or a list of JSON objects.")


def prepare_flight_records(records):
    missing = sorted(
        {
            field
            for record in records
            for field in FLIGHT_REQUIRED_FIELDS
            if field not in record
        }
    )
    if missing:
        raise ValueError(f"Missing required flight field(s): {', '.join(missing)}")

    df = pd.DataFrame(records).copy()

    for column in ["time", "distance"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["time", "distance"]].isnull().any().any():
        raise ValueError("Fields `time` and `distance` must be numeric.")

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
        dates = dates.fillna(pd.to_datetime(df["date"], errors="coerce"))
    else:
        dates = pd.Series([pd.Timestamp(datetime.utcnow().date())] * len(df))

    if dates.isnull().any():
        raise ValueError("Field `date` must be parseable, for example `09/26/2019`.")

    df["flight_month"] = dates.dt.month
    df["flight_day"] = dates.dt.day
    df["flight_dayofweek"] = dates.dt.dayofweek
    df["is_weekend"] = df["flight_dayofweek"].isin([5, 6]).astype(int)

    return df[FLIGHT_FEATURE_COLUMNS]


def _predict_proba_response(model, input_df):
    prediction = model.predict(input_df)
    response = [{"predicted_gender": str(value)} for value in prediction]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)
        classes = [str(label) for label in model.classes_]
        for row, probability in zip(response, probabilities):
            row["probabilities"] = {
                label: round(float(score), 4)
                for label, score in zip(classes, probability)
            }

    return response


def score_hotels(payload, recommender):
    catalog = pd.DataFrame(recommender["catalog"])
    profiles = recommender.get("user_profiles", {})

    top_n = int(payload.get("top_n", 5))
    top_n = max(1, min(top_n, 20))

    user_code = payload.get("userCode", payload.get("user_code"))
    profile = profiles.get(str(user_code), {}) if user_code is not None else {}

    preferred_place = payload.get("place") or profile.get("preferred_place")
    days = payload.get("days", profile.get("avg_days"))
    max_price = payload.get("max_price", payload.get("budget_per_day"))

    if days is not None:
        days = float(days)
    if max_price is not None:
        max_price = float(max_price)

    max_booking_count = max(float(catalog["booking_count"].max()), 1.0)
    catalog["score"] = 0.25 + (catalog["booking_count"] / max_booking_count)

    if preferred_place:
        catalog["score"] += (catalog["place"] == preferred_place).astype(float) * 2.5

    if max_price:
        under_budget = catalog["price"] <= max_price
        catalog["score"] += under_budget.astype(float) * 1.0
        catalog["score"] -= (~under_budget).astype(float) * (
            (catalog["price"] - max_price).clip(lower=0) / max_price
        )

    if days:
        catalog["score"] += (
            1 / (1 + (catalog["avg_days"] - days).abs())
        ) * 0.5

    recommendations = catalog.sort_values(
        ["score", "booking_count", "price"], ascending=[False, False, True]
    ).head(top_n)

    result = []
    for row in recommendations.to_dict(orient="records"):
        item = {
            "hotel_name": row["name"],
            "place": row["place"],
            "price_per_day": round(float(row["price"]), 2),
            "booking_count": int(row["booking_count"]),
            "score": round(float(row["score"]), 4),
        }
        if days:
            item["estimated_total"] = round(float(row["price"]) * days, 2)
        result.append(item)

    return result


def model_status():
    return {
        "flight_price_model": FLIGHT_MODEL_PATH.exists(),
        "gender_classifier": GENDER_MODEL_PATH.exists(),
        "hotel_recommender": HOTEL_RECOMMENDER_PATH.exists(),
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "Voyage Analytics API is running",
            "available_models": model_status(),
            "endpoints": {
                "health": "GET /health",
                "flight_price": "POST /predict/flight",
                "legacy_flight_price": "POST /predict",
                "gender": "POST /predict/gender",
                "hotel_recommendations": "POST /recommend/hotels",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models": model_status()})


@app.route("/predict", methods=["POST"])
@app.route("/predict/flight", methods=["POST"])
def predict_flight_price():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return _json_error("Request body must be valid JSON.")

        records, single = _records_from_payload(payload)
        input_df = prepare_flight_records(records)
        model = _load_artifact("flight", FLIGHT_MODEL_PATH)
        predictions = model.predict(input_df)

        response = [
            {"predicted_flight_price": round(float(value), 2)}
            for value in predictions
        ]
        return jsonify(response[0] if single else response)
    except Exception as exc:
        return _json_error(str(exc))


@app.route("/predict/gender", methods=["POST"])
def predict_gender():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return _json_error("Request body must be valid JSON.")

        records, single = _records_from_payload(payload)
        input_df = pd.DataFrame(records)

        if "age" not in input_df.columns or "company" not in input_df.columns:
            return _json_error("Required fields: `company` and `age`.")
        if "name" not in input_df.columns:
            input_df["name"] = "Unknown"

        input_df["age"] = pd.to_numeric(input_df["age"], errors="coerce")
        if input_df["age"].isnull().any():
            return _json_error("Field `age` must be numeric.")

        model = _load_artifact("gender", GENDER_MODEL_PATH)
        response = _predict_proba_response(model, input_df[["company", "name", "age"]])
        return jsonify(response[0] if single else response)
    except Exception as exc:
        return _json_error(str(exc))


@app.route("/recommend/hotels", methods=["POST"])
def recommend_hotels():
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return _json_error("Request body must be a JSON object.")

        recommender = _load_artifact("hotels", HOTEL_RECOMMENDER_PATH)
        recommendations = score_hotels(payload, recommender)

        return jsonify(
            {
                "recommendations": recommendations,
                "input": payload,
            }
        )
    except Exception as exc:
        return _json_error(str(exc))


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug, use_reloader=False)
