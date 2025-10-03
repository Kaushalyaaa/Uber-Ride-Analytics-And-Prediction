# streamlit_app/utils_fe.py
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple


def load_artifacts(art_dir: str = "artifacts") -> Dict:
    """Load all persisted training artifacts."""
    return {
        "model": joblib.load(f"{art_dir}/final_model.pkl"),
        "imputer": joblib.load(f"{art_dir}/imputer.pkl"),
        "features": joblib.load(f"{art_dir}/features.pkl"),
        "top_pickups": joblib.load(f"{art_dir}/top_pickups.pkl"),
        "top_drops": joblib.load(f"{art_dir}/top_drops.pkl"),
        "vehicle_cols": joblib.load(f"{art_dir}/vehicle_cols.pkl"),
    }


def engineer_features(
    raw: pd.DataFrame, artifacts: Dict, debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the same feature engineering used in training.

    Returns:
      X_ready: DataFrame ready for model.predict_proba (same columns/order as training)
      df:      intermediate engineered frame (for debugging in the UI)
    """
    df = raw.copy()

    # -----------------------------
    # Parse datetime parts
    # -----------------------------
    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    # "Time" may be HH:MM:SS string
    df["Time"] = pd.to_datetime(df.get("Time"), errors="coerce").dt.time
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
    )

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["weekday"].isin([5, 6])

    # -----------------------------
    # Flags / placeholders for possibly-missing columns
    # -----------------------------
    for c in ["Cancelled Rides by Customer", "Cancelled Rides by Driver", "Incomplete Rides"]:
        if c not in df.columns:
            df[c] = np.nan

    df["is_cancelled_customer"] = df["Cancelled Rides by Customer"].notnull()
    df["is_cancelled_driver"] = df["Cancelled Rides by Driver"].notnull()
    df["is_incomplete"] = df["Incomplete Rides"].notnull()

    for c in ["Driver Ratings", "Customer Rating", "Booking Value", "Payment Method"]:
        if c not in df.columns:
            df[c] = np.nan

    df["missing_driver_rating"] = df["Driver Ratings"].isnull()
    df["missing_customer_rating"] = df["Customer Rating"].isnull()
    df["missing_booking_value"] = df["Booking Value"].isnull()
    df["missing_payment_method"] = df["Payment Method"].isnull()

    # Robust fill (as in training) for these two
    for c in ["Avg VTAT", "Avg CTAT"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].fillna(df[c].median())

    # -----------------------------
    # Vehicle one-hot (align to training vehicle columns)
    # -----------------------------
    if "Vehicle Type" not in df.columns:
        df["Vehicle Type"] = "Other"

    dummies_vehicle = pd.get_dummies(df["Vehicle Type"], prefix="vehicle")
    for col in artifacts["vehicle_cols"]:
        if col not in dummies_vehicle.columns:
            dummies_vehicle[col] = 0
    dummies_vehicle = dummies_vehicle[artifacts["vehicle_cols"]]
    df = pd.concat([df, dummies_vehicle], axis=1)

    # -----------------------------
    # Pickup/Drop top-k encoding (+ searchable “Other (type below)” normalization)
    # -----------------------------
    def encode_top_k(series: pd.Series, top_list, prefix: str) -> pd.DataFrame:
        enc = series.where(series.isin(top_list), other="Other")
        dummies = pd.get_dummies(enc, prefix=prefix, drop_first=True)
        return dummies

    if "Pickup Location" not in df.columns:
        df["Pickup Location"] = "Other"
    if "Drop Location" not in df.columns:
        df["Drop Location"] = "Other"

    # normalize the UI placeholder to "Other"
    df["Pickup Location"] = df["Pickup Location"].replace({"Other (type below)": "Other"})
    df["Drop Location"] = df["Drop Location"].replace({"Other (type below)": "Other"})

    d_pickup = encode_top_k(df["Pickup Location"], artifacts["top_pickups"], "pickup")
    d_drop = encode_top_k(df["Drop Location"], artifacts["top_drops"], "drop")
    df = pd.concat([df, d_pickup, d_drop], axis=1)

    # -----------------------------
    # Customer booking frequency
    # -----------------------------
    if "Customer ID" in df.columns:
        counts = df["Customer ID"].value_counts().to_dict()
        df["customer_total_bookings"] = df["Customer ID"].map(counts).fillna(1).astype(int)
    else:
        df["customer_total_bookings"] = 1

    # -----------------------------
    # Drop leaky/raw columns (ensure target/booking status never slip through)
    # -----------------------------
    drop_cols = [
        "Booking ID", "Customer ID", "Pickup Location", "Drop Location",
        "Cancelled Rides by Customer", "Reason for cancelling by Customer",
        "Cancelled Rides by Driver", "Driver Cancellation Reason",
        "Incomplete Rides", "Incomplete Rides Reason",
        "Booking Status", "target_customer_cancelled",
        "Date", "Time", "datetime"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # -----------------------------
    # Final numeric matrix → align to IMPUTER schema → impute
    # -----------------------------
    # Make sure all bools are ints BEFORE selecting numerics (clarity)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    # numeric only
    X_num = df.select_dtypes(include=[np.number]).copy()

    # Prefer imputer's original training schema
    fit_names = getattr(artifacts["imputer"], "feature_names_in_", None)
    if fit_names is not None:
        target_cols = list(fit_names)
    else:
        target_cols = list(artifacts["features"])  # fallback if imputer lacks names

    # EXACT reindex to expected columns; unseen → 0, extras dropped
    X_num = X_num.reindex(columns=target_cols, fill_value=0)

    # Impute & return as DataFrame with the same names
    X_imp = artifacts["imputer"].transform(X_num[target_cols])
    X_ready = pd.DataFrame(X_imp, columns=target_cols)

    # -----------------------------
    # Optional debug
    # -----------------------------
    if debug:
        # How many non-zeros in first row?
        nz = int((X_ready.iloc[0] != 0).sum()) if len(X_ready) else 0
        print(f"[utils_fe] Non-zero features in row 0: {nz}/{X_ready.shape[1]}")
        # Check schema consistency between imputer & saved features (if both available)
        feat_list = artifacts.get("features", [])
        if hasattr(artifacts["imputer"], "feature_names_in_") and feat_list:
            imp_cols = set(artifacts["imputer"].feature_names_in_)
            feat_cols = set(feat_list)
            only_in_imp = imp_cols - feat_cols
            only_in_feat = feat_cols - imp_cols
            if only_in_imp or only_in_feat:
                print("[utils_fe] WARNING: schema mismatch",
                      "\n  in imputer not in features.pkl:", sorted(list(only_in_imp))[:10],
                      "\n  in features.pkl not in imputer:", sorted(list(only_in_feat))[:10])

    return X_ready, df
