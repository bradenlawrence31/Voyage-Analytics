"""
Voyage Analytics - Model Training Script with MLflow Tracking
=============================================================
Run this script ONCE before starting app.py.
It reads your 3 CSV datasets, trains 3 models, saves .pkl files,
and logs all metrics/parameters to MLflow.

Usage:
    python train_models.py

Then view MLflow UI:
    mlflow ui
    Open: http://localhost:5001
"""

from pathlib import Path
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "datasets"
MODELS_DIR  = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FLIGHTS_CSV = DATA_DIR / "flights.csv"
HOTELS_CSV  = DATA_DIR / "hotels.csv"
USERS_CSV   = DATA_DIR / "users.csv"

# ── MLflow Setup ───────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = BASE_DIR / "mlflow"
mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
EXPERIMENT_NAME = "Voyage Analytics"

mlflow.set_experiment(EXPERIMENT_NAME)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FLIGHT PRICE REGRESSION MODEL
# ══════════════════════════════════════════════════════════════════════════════
def train_flight_model():
    print("\n[1/3] Training Flight Price Regression Model...")

    df = pd.read_csv(FLIGHTS_CSV)

    # Parse date and extract features
    dates = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["flight_month"]     = dates.dt.month
    df["flight_day"]       = dates.dt.day
    df["flight_dayofweek"] = dates.dt.dayofweek
    df["is_weekend"]       = df["flight_dayofweek"].isin([5, 6]).astype(int)

    FEATURES = ["from", "to", "flightType", "time", "distance", "agency",
                "flight_month", "flight_day", "flight_dayofweek", "is_weekend"]
    TARGET   = "price"

    X = df[FEATURES]
    y = df[TARGET]

    cat_cols = ["from", "to", "flightType", "agency"]
    num_cols = ["time", "distance", "flight_month", "flight_day",
                "flight_dayofweek", "is_weekend"]

    # Model parameters
    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42
    }

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(**params))
    ])

    # ── MLflow Run ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="Flight Price Regression"):

        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        # Metrics
        mae  = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        mape = np.mean(np.abs((y - preds) / y)) * 100

        # Log parameters
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("training_rows", len(df))

        # Log metrics
        mlflow.log_metric("mae", round(mae, 4))
        mlflow.log_metric("rmse", round(rmse, 4))
        mlflow.log_metric("mape", round(mape, 4))

        # Log model
        mlflow.sklearn.log_model(pipeline, "flight_price_model")

        print(f"    ✔ Flight model trained | MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%")
        print(f"    ✔ MLflow Run ID: {mlflow.active_run().info.run_id}")

    out_path = MODELS_DIR / "flight_price_model.pkl"
    joblib.dump(pipeline, out_path)
    print(f"    ✔ Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GENDER CLASSIFICATION MODEL
# ══════════════════════════════════════════════════════════════════════════════
def train_gender_model():
    print("\n[2/3] Training Gender Classification Model...")

    df = pd.read_csv(USERS_CSV)

    FEATURES = ["company", "name", "age"]
    TARGET   = "gender"

    X = df[FEATURES]
    y = df[TARGET]

    cat_cols = ["company", "name"]
    num_cols = ["age"]

    # Model parameters
    params = {
        "n_estimators": 100,
        "random_state": 42
    }

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(**params))
    ])

    # ── MLflow Run ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="Gender Classification"):

        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        # Metrics
        accuracy = accuracy_score(y, preds)
        f1       = f1_score(y, preds, average="weighted")

        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("training_rows", len(df))
        mlflow.log_param("classes", list(df[TARGET].unique()))

        # Log metrics
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("f1_score", round(f1, 4))

        # Log model
        mlflow.sklearn.log_model(pipeline, "gender_classifier")

        print(f"    ✔ Gender model trained | Accuracy: {accuracy*100:.1f}% | F1 Score: {f1:.4f}")
        print(f"    ✔ MLflow Run ID: {mlflow.active_run().info.run_id}")

    out_path = MODELS_DIR / "gender_classifier.pkl"
    joblib.dump(pipeline, out_path)
    print(f"    ✔ Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. HOTEL RECOMMENDATION MODEL
# ══════════════════════════════════════════════════════════════════════════════
def train_hotel_recommender():
    print("\n[3/3] Building Hotel Recommendation Model...")

    hotels_df = pd.read_csv(HOTELS_CSV)
    users_df  = pd.read_csv(USERS_CSV)

    # Build hotel catalog
    catalog = (
        hotels_df.groupby(["name", "place"])
        .agg(
            price        =("price", "mean"),
            avg_days     =("days",  "mean"),
            booking_count=("travelCode", "count"),
        )
        .reset_index()
    )

    # Build user profiles
    user_hotels = hotels_df.merge(users_df, left_on="userCode", right_on="code", how="left")

    user_profiles = {}
    for user_code, group in user_hotels.groupby("userCode"):
        preferred_place = group["place"].mode()[0] if not group["place"].empty else None
        avg_days        = float(group["days"].mean()) if not group["days"].empty else None
        budget_per_day  = float(group["price"].mean()) if not group["price"].empty else None

        user_profiles[str(user_code)] = {
            "preferred_place": preferred_place,
            "avg_days":        avg_days,
            "budget_per_day":  budget_per_day,
        }

    recommender = {
        "catalog":       catalog.to_dict(orient="records"),
        "user_profiles": user_profiles,
    }

    # ── MLflow Run ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="Hotel Recommender"):

        # Log parameters
        mlflow.log_param("model_type", "Scoring-Based Recommender")
        mlflow.log_param("total_hotels", len(catalog))
        mlflow.log_param("total_places", int(catalog["place"].nunique()))
        mlflow.log_param("total_users_profiled", len(user_profiles))

        # Log metrics
        mlflow.log_metric("num_hotels", len(catalog))
        mlflow.log_metric("num_places", int(catalog["place"].nunique()))
        mlflow.log_metric("num_user_profiles", len(user_profiles))
        mlflow.log_metric("avg_price_per_day", round(float(catalog["price"].mean()), 2))
        mlflow.log_metric("avg_bookings_per_hotel", round(float(catalog["booking_count"].mean()), 2))

        print(f"    ✔ Hotel catalog built | {len(catalog)} hotels across {catalog['place'].nunique()} places")
        print(f"    ✔ User profiles built | {len(user_profiles)} users profiled")
        print(f"    ✔ MLflow Run ID: {mlflow.active_run().info.run_id}")

    out_path = MODELS_DIR / "hotel_recommender.pkl"
    joblib.dump(recommender, out_path)
    print(f"    ✔ Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Voyage Analytics — Training All Models with MLflow")
    print("=" * 60)
    print(f"  MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print("=" * 60)

    train_flight_model()
    train_gender_model()
    train_hotel_recommender()

    print("\n" + "=" * 60)
    print("  All models trained and saved to /models folder!")
    print("  MLflow runs logged successfully!")
    print()
    print("  To view MLflow UI run:")
    print("  mlflow ui --port 5001")
    print("  Then open: http://localhost:5001")
    print("=" * 60)