# streamlit_app/app.py
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils_fe import load_artifacts, engineer_features

# ------------------ Page & Theme ------------------
st.set_page_config(page_title="Uber-style Cancellation Risk", page_icon="🚖", layout="wide")

# Softer colors; lighter black; no heavy gradients; car has its own lane.
st.markdown("""
<style>
:root{
  --uber-green:#19C37D;
  --bg:#111111;            /* lighter black */
  --card:#1a1a1a;          /* slightly lighter than bg */
  --muted:#b8b8b8;         /* soft gray text */
  --text:#d8d8d8;          /* main text color (not bright white) */
  --line:#262626;
}
.stApp{ background: var(--bg); }
h1,h2,h3,h4,h5,h6{ color: var(--text); }
p, .small, label{ color: var(--muted) !important; }
[data-testid="stMetricValue"]{ color: var(--uber-green); }

/* Cards */
.card{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: none;
}

/* Buttons */
.stButton>button{
  background: var(--uber-green);
  border: none;
  color: #0b0b0b;
  border-radius: 12px;
  font-weight: 700;
  padding: 8px 16px;
}
.stButton>button:hover{ filter: brightness(1.07); }

/* Inputs */
.stTextInput>div>div>input, .stNumberInput input, .stTimeInput input, .stDateInput input{
  background: #1f1f1f; color: var(--text);
  border: 1px solid var(--line); border-radius: 10px;
}
.stSelectbox div[data-baseweb="select"]>div{
  background: #1f1f1f; color: var(--text);
  border: 1px solid var(--line); border-radius: 10px;
}

/* Header (no gradient box; simple title + separate road) */
.header-title { padding: 6px 12px; }
.roadwrap { position: relative; height: 56px; margin-top: 6px; }
.road {
  position:absolute; left:0; right:0; top:26px; height:4px;
  background: repeating-linear-gradient(90deg, rgba(255,255,255,0.12) 0 24px, transparent 24px 48px);
}
.car {
  position:absolute; top:0; left:-80px; font-size:40px;
  animation: drive 7s linear infinite;
}
@keyframes drive {
  0% { left:-80px; }
  100% { left: calc(100% + 80px); }
}

/* Tidy tabs underline to green */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { border-color: var(--uber-green); }
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
left, mid = st.columns([1,6])
with left:
  logo_path = os.path.join("assets","uber_logo.png")
  if os.path.exists(logo_path):
    st.image(logo_path, use_container_width=True)
  else:
    st.write("🟩 Place logo at `assets/uber_logo.png`")

with mid:
  st.markdown("<div class='header-title'><h3>Uber-style Risk Console</h3>"
              "<p class='small'>Predict customer cancellation in real-time or for CSV batches. "
              "Tune the decision threshold in the sidebar.</p></div>", unsafe_allow_html=True)
  st.markdown("<div class='roadwrap'><div class='car'>🚕</div><div class='road'></div></div>", unsafe_allow_html=True)

# ------------------ Load artifacts ------------------
ART = load_artifacts("artifacts")
MODEL = ART["model"]
TOP_PICKS = list(ART["top_pickups"])
TOP_DROPS = list(ART["top_drops"])
VEH_TYPES = ["Sedan","SUV","Hatchback","Mini","Other"]

# Session state history for visuals (store dicts with ts, prob, pred, y_true)
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Decision threshold (prob ≥ threshold ⇒ Cancelled)", 0.05, 0.95, 0.50, 0.01)
st.sidebar.caption("Lower threshold → higher recall (catch more cancellations). Higher threshold → higher precision (fewer false alarms).")

tab1, tab2 = st.tabs(["🧍 Single Ride", "📄 Batch CSV"])

# ===================== SINGLE RIDE =====================
with tab1:
    st.subheader("🧍 Single Ride – Fill the details ↪")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        vehicle = st.selectbox("🚗 Vehicle type", options=VEH_TYPES, index=0)
        driver_rating = st.number_input("⭐ Driver rating (0–5)", 0.0, 5.0, 4.5, 0.1)
        cust_rating   = st.number_input("🙂 Customer rating (0–5)", 0.0, 5.0, 4.2, 0.1)
        booking_value = st.number_input("💵 Fare estimate", 0.0, value=250.0, step=10.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        date = st.date_input("📅 Ride date")
        time_inp = st.time_input("🕒 Ride time")
        vehicle_eta   = st.number_input("🚕 Vehicle ETA to pickup (min)", 0.0, value=5.0, step=0.5)   # Avg VTAT
        customer_walk = st.number_input("🚶 Customer time to pickup (min)", 0.0, value=5.0, step=0.5) # Avg CTAT
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        pickup_choice = st.selectbox("📍 Pickup location (searchable)",
                                     options=["Other (type below)"] + sorted(TOP_PICKS))
        pickup_text = st.text_input("…or type pickup", value="" if pickup_choice!="Other (type below)" else "")
        drop_choice = st.selectbox("🏁 Drop location (searchable)",
                                   options=["Other (type below)"] + sorted(TOP_DROPS))
        drop_text = st.text_input("…or type drop", value="" if drop_choice!="Other (type below)" else "")
        payment_method = st.selectbox("💳 Payment method", ["Cash","Card","Wallet","UPI","Other"])
        customer_id = st.text_input("🧾 Customer ID (optional)", value="")
        st.markdown("</div>", unsafe_allow_html=True)

    # Final values; require something non-empty
    pickup_final = pickup_text.strip() if pickup_choice == "Other (type below)" and pickup_text.strip() else pickup_choice
    drop_final   = drop_text.strip()   if drop_choice   == "Other (type below)" and drop_text.strip()   else drop_choice

    run = st.button("🟢 Predict cancellation risk")
    if run:
        # validate required fields
        if not pickup_final or pickup_final == "Other (type below)":
            st.error("Please select or type a **Pickup location**.")
            st.stop()
        if not drop_final or drop_final == "Other (type below)":
            st.error("Please select or type a **Drop location**.")
            st.stop()

        raw = pd.DataFrame([{
            "Vehicle Type": vehicle,
            "Driver Ratings": driver_rating,
            "Customer Rating": cust_rating,
            "Booking Value": booking_value,
            "Date": pd.to_datetime(date),
            "Time": pd.to_datetime(str(time_inp)),
            "Avg VTAT": vehicle_eta,
            "Avg CTAT": customer_walk,
            "Pickup Location": pickup_final,
            "Drop Location": drop_final,
            "Payment Method": payment_method,
            "Customer ID": customer_id if customer_id else np.nan,
        }])

        X_ready, _ = engineer_features(raw, ART)

        # DEBUG: show how many non-zero features we have
        nonzero = int((X_ready.iloc[0] != 0).sum())
        st.caption(f"Non-zero features in this row: **{nonzero} / {X_ready.shape[1]}**")
        if nonzero < 3:
            st.warning("Almost all features are zero. This usually means your **artifacts are out of sync** with the feature engineering. "
                       "Re-export imputer + features + model from the same training run and copy them to `streamlit_app/artifacts/`.")

        prob = float(MODEL.predict_proba(X_ready)[:, 1])
        pred = int(prob >= threshold)

        m1, m2, m3 = st.columns(3)
        m1.metric("🎯 Probability of cancellation", f"{prob:.3f}")
        m2.metric("⚖️ Threshold", f"{threshold:.2f}")
        m3.metric("✅ Decision", "Cancelled" if pred==1 else "Not cancelled")

        # Persist the index so saving a label won't “lose” the row on rerun
        idx = len(st.session_state.history)
        st.session_state.history.append({"ts": time.time(), "prob": prob, "pred": pred, "y_true": None})

        with st.expander("✍️ Mark actual outcome (optional) for visuals"):
            choice = st.radio("Actual outcome for THIS ride?", ["—", "Not cancelled (0)", "Cancelled (1)"],
                              horizontal=True, key=f"label_choice_{idx}")
            if st.button("Save actual outcome", key=f"save_{idx}"):
                if choice == "Not cancelled (0)":
                    st.session_state.history[idx]["y_true"] = 0
                    st.success("Saved actual outcome: 0 (Not cancelled)")
                elif choice == "Cancelled (1)":
                    st.session_state.history[idx]["y_true"] = 1
                    st.success("Saved actual outcome: 1 (Cancelled)")
                else:
                    st.info("No label saved.")


    # ---------- Visuals: show once you have labels ----------
    labeled = [h for h in st.session_state.history if h["y_true"] is not None]
    if len(labeled) >= 2:
        last10 = labeled[-10:]
        y_true = np.array([x["y_true"] for x in last10])
        y_pred = np.array([x["pred"] for x in last10])

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        v1, v2, v3 = st.columns([1,1,2])

        # CM thumbnail
        with v1:
            st.markdown("##### 📊 Confusion matrix")
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            im = ax.imshow(cm, cmap="Greens")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_xticklabels(['Not', 'Canc']); ax.set_yticklabels(['Not', 'Canc'])
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha='center', va='center', color='black')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

        # TP/FP/TN/FN bars
        with v2:
            st.markdown("##### 📈 Last 10: counts")
            labels = ["TP","FP","TN","FN"]
            vals = [tp, fp, tn, fn]
            fig2, ax2 = plt.subplots(figsize=(3.2, 3.0))
            ax2.bar(labels, vals)
            for i, v in enumerate(vals):
                ax2.text(i, v + 0.05, str(v), ha='center', color=("#0b0b0b" if v>0 else "#d8d8d8"))
            ax2.set_ylim(0, max(vals)+1 if max(vals) > 0 else 1)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=False)

        # Probability sparkline (last 20 preds)
        with v3:
            st.markdown("##### 🔎 Probability history (last 20)")
            recent = st.session_state.history[-20:]
            probs = np.array([h["prob"] for h in recent])
            xs = np.arange(1, len(probs)+1)
            fig3, ax3 = plt.subplots(figsize=(6.0, 3.0))
            ax3.plot(xs, probs, marker='o')
            ax3.axhline(y=threshold, linestyle='--', linewidth=1)
            ax3.set_xlabel("Prediction # (most recent on right)")
            ax3.set_ylabel("P(cancel)")
            ax3.set_ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
    else:
        st.caption("Tip: label a few predictions to unlock the visuals (CM, bars, and sparkline).")

# ===================== BATCH CSV =====================
with tab2:
    st.subheader("📄 Batch CSV Prediction")
    st.caption("Columns: Date, Time, Vehicle Type, Pickup Location, Drop Location, Driver Ratings, Customer Rating, Booking Value, Avg VTAT, Avg CTAT, Payment Method, Customer ID.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df_raw = pd.read_csv(up)

        # Map friendly names, if present
        rename_map = {
            "Vehicle ETA to pickup (min)": "Avg VTAT",
            "Customer time to pickup (min)": "Avg CTAT",
        }
        for k, v in rename_map.items():
            if k in df_raw.columns and v not in df_raw.columns:
                df_raw[v] = df_raw[k]

        X_ready, _ = engineer_features(df_raw, ART)
        probs = ART["model"].predict_proba(X_ready)[:, 1]
        preds = (probs >= threshold).astype(int)

        out = df_raw.copy()
        out["prob_cancel"] = probs
        out["predicted_cancel"] = preds

        st.success(f"Predictions completed for {len(out)} rides.")
        st.dataframe(out.head(50), use_container_width=True)
        st.download_button("⬇️ Download predictions",
                           out.to_csv(index=False).encode("utf-8"),
                           "predictions.csv", "text/csv")
