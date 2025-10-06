import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, base64
from pathlib import Path
from datetime import datetime, time as dtime

# ========================= Page setup =========================
st.set_page_config(
    page_title="PreCancello – Ride Cancellation Predictor",
    page_icon="🚗",
    layout="wide"
)

# --------------- Uber theme (CSS) & UX polish ---------------
UBER_BLACK = "#000000"
UBER_YELLOW = "#FFC043"
UBER_DARK = "#111111"
UBER_CARD = "#161616"
UBER_TEXT = "#EAEAEA"
UBER_MUTED = "#B0B0B0"

THEME_CSS = f"""
<style>
/* Push content below Streamlit's top bar (where Deploy sits) */
.block-container {{
  padding-top: 2.8rem !important;
}}

/* Global background + text */
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, {UBER_BLACK} 0%, {UBER_DARK} 100%);
  color: {UBER_TEXT};
}}

/* Sidebar */
[data-testid="stSidebar"] > div:first-child {{
  background: {UBER_BLACK};
  border-right: 1px solid #222;
  color: {UBER_TEXT};
}}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{
  color: {UBER_TEXT};
}}

/* Top toolbar (Deploy bar) -> black */
header[data-testid="stHeader"] {{
  background: #000 !important; color: #fff !important; border-bottom: 1px solid #222 !important;
}}
header[data-testid="stHeader"] > div {{ background: #000 !important; }}
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] [data-testid="baseButton-header"] {{ color: #fff !important; }}
header[data-testid="stHeader"] svg path,
header[data-testid="stHeader"] svg circle,
header[data-testid="stHeader"] svg rect {{ fill:#fff !important; stroke:#fff !important; }}

/* Headings */
h1, h2, h3, h4, h5 {{ color: {UBER_TEXT}; }}

/* Header (logo + chip + title) */
.header-bar {{
  display:flex; align-items:center; gap:12px; margin:6px 0 6px 0;
}}
.header-logo {{
  width:150px; height:150px; border-radius:8px; object-fit:cover; 
}}
.logo-chip {{
  background:{UBER_YELLOW}; color:#000; padding:6px 10px; border-radius:10px; font-weight:900;
}}
.brand-title {{ font-weight:900; font-size:1.7rem; letter-spacing:0.3px; }}
.brand-sub {{ color:{UBER_MUTED}; margin-left:6px; font-size:0.95rem; }}

/* Enhanced 3D Road Scene with Animated Elements */
.road {{
  position: relative;
  height: 60px;                /* taller for 3D scene */
  background: linear-gradient(180deg, #1a1a1a 0%, #2b2b2b 50%, #1a1a1a 100%);
  border-radius: 8px;
  overflow: hidden;
  margin: 6px 0 20px 0;
  box-shadow: inset 0 2px 6px rgba(0,0,0,.35);
}}

/* Road markings */
.road:before {{
  content: "";
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 2px;
  background: repeating-linear-gradient(
    90deg,
    #ffd700 0px,
    #ffd700 20px,
    transparent 20px,
    transparent 40px
  );
  animation: roadMove 2s linear infinite;
}}

@keyframes roadMove {{
  0% {{ transform: translateX(0); }}
  100% {{ transform: translateX(40px); }}
}}

/* 3D People standing on sidewalk */
.road .person {{
  position: absolute;
  bottom: 0;
  width: 8px;
  height: 20px;
  background: linear-gradient(180deg, #ff6b6b 0%, #4ecdc4 50%, #45b7d1 100%);
  border-radius: 4px 4px 2px 2px;
  animation: personSway 3s ease-in-out infinite;
}}

.road .person:before {{
  content: "";
  position: absolute;
  top: -3px;
  left: 1px;
  width: 6px;
  height: 6px;
  background: #ffdbac;
  border-radius: 50%;
}}

.road .person:nth-child(1) {{ left: 15%; animation-delay: 0s; }}
.road .person:nth-child(2) {{ left: 35%; animation-delay: 1s; }}
.road .person:nth-child(3) {{ left: 65%; animation-delay: 0.5s; }}
.road .person:nth-child(4) {{ left: 85%; animation-delay: 1.5s; }}

@keyframes personSway {{
  0%, 100% {{ transform: rotate(-1deg); }}
  50% {{ transform: rotate(1deg); }}
}}

/* 3D Trees */
.road .tree {{
  position: absolute;
  bottom: 0;
  width: 12px;
  height: 25px;
  background: linear-gradient(180deg, #8b4513 0%, #8b4513 40%, #228b22 40%, #228b22 100%);
  border-radius: 2px;
  animation: treeSway 4s ease-in-out infinite;
}}

.road .tree:before {{
  content: "";
  position: absolute;
  top: -8px;
  left: -2px;
  width: 16px;
  height: 16px;
  background: radial-gradient(circle, #32cd32 0%, #228b22 70%);
  border-radius: 50%;
}}

.road .tree:nth-child(5) {{ left: 5%; animation-delay: 0s; }}
.road .tree:nth-child(6) {{ left: 25%; animation-delay: 1.2s; }}
.road .tree:nth-child(7) {{ left: 75%; animation-delay: 0.8s; }}
.road .tree:nth-child(8) {{ left: 95%; animation-delay: 2s; }}

@keyframes treeSway {{
  0%, 100% {{ transform: rotate(-2deg); }}
  50% {{ transform: rotate(2deg); }}
}}

/* Traffic Lights */
.road .traffic-light {{
  position: absolute;
  bottom: 0;
  left: 50%;
  width: 6px;
  height: 30px;
  background: #333;
  border-radius: 3px;
  animation: trafficBlink 3s ease-in-out infinite;
}}

.road .traffic-light:before {{
  content: "";
  position: absolute;
  top: 2px;
  left: 1px;
  width: 4px;
  height: 4px;
  background: #ff0000;
  border-radius: 50%;
  box-shadow: 0 6px 0 #ffff00, 0 12px 0 #00ff00;
}}

@keyframes trafficBlink {{
  0%, 30% {{ opacity: 1; }}
  31%, 100% {{ opacity: 0.3; }}
}}

/* Enhanced 3D Car */
.car {{
  position: absolute;
  top: 20px;
  left: -60px;
  width: 50px;
  height: 18px;
  background: linear-gradient(180deg, #ff6b6b 0%, #4ecdc4 50%, #45b7d1 100%);
  border-radius: 8px 12px 6px 6px;
  box-shadow: 0 4px 12px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.2);
  animation: drive 6s linear infinite, carBounce 2s ease-in-out infinite;
  transform-style: preserve-3d;
}}

/* Car roof/windshield */
.car:before {{
  content: "";
  position: absolute;
  top: -6px;
  left: 8px;
  right: 12px;
  height: 8px;
  background: linear-gradient(180deg, #e8f4f8 0%, #b8e6f1 100%);
  border-radius: 10px 12px 3px 3px;
  box-shadow: inset 0 0 6px rgba(0,0,0,.3);
}}

/* Car windows */
.car .win {{
  position: absolute;
  top: 1px;
  width: 12px;
  height: 9px;
  background: linear-gradient(180deg, #e8f4f8 0%, #b8e6f1 100%);
  border-radius: 3px;
  box-shadow: inset 0 -1px 3px rgba(0,0,0,.2);
  opacity: 0.9;
}}
.car .w1 {{ left: 7px; }}
.car .w2 {{ left: 21px; }}
.car .w3 {{ left: 34px; width: 9px; border-top-right-radius: 8px; }}

/* Enhanced wheels with rotation */
.car .wh {{
  position: absolute;
  bottom: -6px;
  width: 10px;
  height: 10px;
  background: radial-gradient(circle, #1a1a1a 0%, #333 50%, #1a1a1a 100%);
  border-radius: 50%;
  box-shadow: inset 0 -2px 0 rgba(255,255,255,.1);
  animation: wheelSpin 0.5s linear infinite;
}}
.car .wl {{ left: 8px; }}
.car .wr {{ right: 8px; }}

@keyframes wheelSpin {{
  0% {{ transform: rotate(0deg); }}
  100% {{ transform: rotate(360deg); }}
}}

/* Enhanced headlights */
.car:after {{
  content: "";
  position: absolute;
  right: -12px;
  top: 5px;
  width: 12px;
  height: 6px;
  background: radial-gradient(ellipse at left, rgba(255,255,200,.9) 0%, rgba(255,255,200,0) 80%);
  filter: blur(2px);
  opacity: 0.9;
  animation: headlightFlicker 4s ease-in-out infinite;
}}

@keyframes headlightFlicker {{
  0%, 90%, 100% {{ opacity: 0.9; }}
  95% {{ opacity: 0.6; }}
}}

/* Car shadow */
.car .shadow {{
  position: absolute;
  bottom: -8px;
  left: 2px;
  right: 2px;
  height: 4px;
  background: radial-gradient(ellipse, rgba(0,0,0,.3) 0%, transparent 70%);
  border-radius: 50%;
  animation: shadowMove 6s linear infinite;
}}

@keyframes shadowMove {{
  0% {{ left: 2px; right: 2px; }}
  50% {{ left: 1px; right: 3px; }}
  100% {{ left: 2px; right: 2px; }}
}}

@keyframes drive {{ 
  0% {{ left: -60px; }} 
  100% {{ left: calc(100% + 60px); }} 
}}

@keyframes carBounce {{ 
  0%, 100% {{ transform: translateY(0) rotateX(0deg); }} 
  50% {{ transform: translateY(-2px) rotateX(1deg); }} 
}}

/* Inputs readable */
label, .stSelectbox label, .stNumberInput label, .stDateInput label, .stTimeInput label {{
  color: {UBER_TEXT} !important; font-weight: 600;
}}
/* Remove yellow hover boxes */
.stSelectbox div[role="combobox"]:hover, .stNumberInput input:hover, .stDateInput:hover, .stTimeInput:hover {{
  outline: none;
  border-color: #444;
}}

/* Buttons */
div.stButton > button {{
  background-color: {UBER_YELLOW}; color: #000; border-radius: 12px; border: 0;
  font-weight: 800; padding: 8px 16px; transition: transform .06s, box-shadow .12s;
}}
div.stButton > button:hover {{
  transform: translateY(-1px); box-shadow: 0 8px 18px rgba(255,192,67,.18);
}}

/* Metrics: ensure value is bright/visible */
div[data-testid="stMetric"] {{
  background: {UBER_CARD}; border: 1px solid #2a2a2a; border-radius: 14px; padding: 12px 14px;
  transition: transform .08s, box-shadow .12s;
}}
div[data-testid="stMetric"]:hover {{ transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,.35); }}
div[data-testid="stMetricValue"] {{ color: {UBER_TEXT} !important; font-weight: 800 !important; }}

/* Expander */
.streamlit-expanderHeader {{
  background: {UBER_CARD}; color: {UBER_TEXT}; border-radius: 10px; border: 1px solid #2a2a2a;
}}

/* Dataframe */
div[data-testid="stDataFrame"] {{ border: 1px solid #2a2a2a; border-radius: 12px; }}

/* Risk band chip (color per band) */
.risk-chip {{
  display:inline-block; padding:4px 10px; border-radius:999px; font-weight:800; letter-spacing:.2px;
}}
.risk-high {{ background:#2a0000; color:#ff6b6b; border:1px solid #661b1b; }}
.risk-med  {{ background:#2a2300; color:#ffd166; border:1px solid #665b1b; }}
.risk-low  {{ background:#032314; color:#95d5b2; border:1px solid #1b6647; }}

/* Enhanced Visual Elements */

/* Better spacing for content sections */
.content-section {{
  margin: 20px 0;
  padding: 15px 0;
}}

/* Improved spacing for metrics */
.metrics-container {{
  margin: 15px 0 20px 0;
  padding: 12px 0;
  background: rgba(22, 22, 22, 0.3);
  border-radius: 12px;
  border: 1px solid #333;
}}

/* Enhanced visual elements */
.risk-chip {{
  display: inline-block;
  padding: 6px 12px;
  border-radius: 20px;
  font-weight: 700;
  font-size: 0.9rem;
  margin: 10px 0;
  box-shadow: 0 2px 8px rgba(0,0,0,.2);
}}

/* Smooth transitions for all interactive elements */
* {{
  transition: all 0.2s ease;
}}

/* Enhanced button hover effects */
div.stButton > button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(255,192,67,.3);
}}

/* Enhanced Expander Styling */
.streamlit-expanderHeader {{
  background: {UBER_CARD} !important;
  color: {UBER_TEXT} !important;
  border-radius: 10px !important;
  border: 1px solid #2a2a2a !important;
  font-weight: 600 !important;
}}

.streamlit-expanderContent {{
  background: transparent !important;
  border: none !important;
}}

/* Footer spacing */
.footer-note {{ 
  margin-top: 60px;  /* Increased spacing */
  padding: 20px 0;
  border-top: 1px solid #333;
}}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ----------------- Artifacts -----------------
ART = Path(__file__).parent / "artifacts"
MODEL_PATH = ART / "final_model.pkl"
FEAT_PATH = ART / "features.pkl"
META_PATH = ART / "meta.json"
LOGO_PATH = ART / "logo.png"  # small logo (48–64 px)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    features = list(joblib.load(FEAT_PATH))
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
        except Exception:
            meta = {}
    return model, features, meta


model, FEATURES, META = load_artifacts()
FEATURES_SET = set(FEATURES)

default_threshold = float(META.get("suggested_threshold", 0.35))
base_rate = META.get("base_rate", None)
calibrated = bool(META.get("calibrated", False))


# ----------------- Header (logo + chip + title + nicer car) -----------------
def _logo_base64(p: Path) -> str | None:
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode("utf-8")
    return None


with st.container():
    cols = st.columns([1, 7, 1])
    with cols[1]:
        b64 = _logo_base64(LOGO_PATH)
        logo_img = f'<img class="header-logo" src="data:image/png;base64,{b64}"/>' if b64 else ""
        st.markdown(
            f"""
            <div class="header-bar">
              {logo_img}
              <span class="logo-chip">Pre</span>
              <span class="brand-title">Cancello</span>
            </div>
            <div class="road">
              <!-- 3D People -->
              <div class="person"></div>
              <div class="person"></div>
              <div class="person"></div>
              <div class="person"></div>

              <!-- 3D Trees -->
              <div class="tree"></div>
              <div class="tree"></div>
              <div class="tree"></div>
              <div class="tree"></div>

              <!-- Traffic Light -->
              <div class="traffic-light"></div>

              <!-- Enhanced 3D Car -->
              <div class="car">
                <div class="win w1"></div>
                <div class="win w2"></div>
                <div class="win w3"></div>
                <div class="wh wl"></div>
                <div class="wh wr"></div>
                <div class="shadow"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------- One-hot groups from schema -----------------
VEH_PREFIX, PAY_PREFIX, PICK_PREFIX, DROP_PREFIX = "vehicle_", "payment_", "pickup_", "drop_"
VEH_COLS = [c for c in FEATURES if c.startswith(VEH_PREFIX)]
PAY_COLS = [c for c in FEATURES if c.startswith(PAY_PREFIX)]
PICK_COLS = [c for c in FEATURES if c.startswith(PICK_PREFIX)]
DROP_COLS = [c for c in FEATURES if c.startswith(DROP_PREFIX)]


def labels_from_onehot(cols, prefix): return [c[len(prefix):] for c in cols]


VEH_LABELS, PAY_LABELS, PICK_LABELS, DROP_LABELS = (
    labels_from_onehot(VEH_COLS, VEH_PREFIX),
    labels_from_onehot(PAY_COLS, PAY_PREFIX),
    labels_from_onehot(PICK_COLS, PICK_PREFIX),
    labels_from_onehot(DROP_COLS, DROP_PREFIX),
)


def onehot_name(prefix, label): return f"{prefix}{label}"


def set_onehot_group(row_dict, labels, prefix, chosen_label):
    if chosen_label not in labels:
        if "Other" in labels:
            chosen_label = "Other"
        elif labels:
            chosen_label = labels[0]
        else:
            return
    for lab in labels:
        row_dict[f"{prefix}{lab}"] = (lab == chosen_label)


def build_feature_row(values_dict):
    return pd.DataFrame([{c: values_dict.get(c, np.nan) for c in FEATURES}])


# ----------------- Risk & Action helpers -----------------
def risk_band(prob: float, base_rate_val: float | None, thr: float) -> str:
    br = base_rate_val if base_rate_val is not None else 0.07
    if prob >= max(thr, 0.25): return "High"
    if prob >= max(0.10, 2 * br): return "Medium"
    return "Low"


def action_suggestions(band: str) -> list[str]:
    # Demo suggestions – company sets policy in production
    if band == "High":
        return [
            "Reconfirm rider & driver now",
            "Prefer closest driver (cut travel time)",
            "Offer small driver incentive / priority dispatch",
            "Use firm payment (hold/UPI confirm) if policy allows",
        ]
    if band == "Medium":
        return [
            "Send soft reconfirmation",
            "Avoid long pickups; prefer popular areas",
            "Monitor until dispatch",
        ]
    return [
        "No extra action needed",
        "Proceed with normal assignment",
    ]


def expected_cost_if(flag: bool, p: float, c_fp: float, c_fn: float, c_int: float) -> float:
    if flag:  # act: pay intervention + risk of false alarm
        return c_int + (1.0 - p) * c_fp
    else:  # no act: risk of a cancel
        return p * c_fn


# ----------------- Sidebar -----------------
cap = "Calibrated " if calibrated else ""
st.sidebar.markdown("### ⚙️ Settings")
st.sidebar.caption(f" ")

threshold = st.sidebar.slider("🎯 Decision threshold (Cancel if P ≥ threshold)",
                              0.00, 0.95, float(default_threshold), 0.01)

if base_rate is not None:
    st.sidebar.caption(f"📊 Baseline cancel rate: **{float(base_rate):.1%}**")

# Collapsible, simple cost guide (non-technical labels)
with st.sidebar.expander("💸 Simple cost guide (optional)", expanded=False):
    c_fp = st.number_input("Cost of a false alarm", min_value=0.0, value=1.0, step=0.5,
                           help="When we flag but it wouldn’t cancel")
    c_fn = st.number_input("Cost of missing a cancel", min_value=0.0, value=4.0, step=0.5,
                           help="When we don’t flag but it cancels")
    c_int = st.number_input("Cost to take action", min_value=0.0, value=0.5, step=0.5, help="Message, incentives, etc.")

mode = st.sidebar.radio("🧪 Mode", ["Single Prediction", "Batch Prediction (CSV)"], index=0)
DEFAULT_TIME = dtime(10, 0)

# Enhanced Help Section - Diary Style with Streamlit Integration
help_expanded = st.expander("❓ How to use this app", expanded=False)
with help_expanded:
    st.markdown("""
    <div style="background: #161616; padding: 20px; border-radius: 12px; border: 1px solid #2a2a2a;">
      <h3 style="color: #FFC043; margin-top: 0;">📖 How to Use PreCancello</h3>
      <div style="color: #EAEAEA; line-height: 1.6;">
        <p><strong style="color: #FFC043;">Step 1:</strong> Pick booking date & time, set VTAT/CTAT, and choose Vehicle / Payment / Pickup / Drop.</p>
        <p><strong style="color: #FFC043;">Step 2:</strong> Click <strong>Predict</strong>. You'll see the probability of cancellation.</p>
        <p><strong style="color: #FFC043;">Step 3:</strong> The <strong>Risk band</strong> color and <strong>Recommended action</strong> explain what to do.</p>
        <p><strong style="color: #FFC043;">Step 4:</strong> Use the slider in the sidebar to change the <strong>decision threshold</strong>.</p>
        <p><strong style="color: #FFC043;">Step 5:</strong> (Optional) Open the <strong>Simple cost guide</strong> to reflect the business trade-off.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------- Time features -----------------
def derive_time_features(date_val, time_val):
    dt = datetime(date_val.year, date_val.month, date_val.day, time_val.hour, time_val.minute)
    return {"hour": int(dt.hour), "weekday": int(dt.weekday()), "month": int(dt.month),
            "is_weekend": bool(dt.weekday() >= 5)}


# ========================= Single Prediction =========================
if mode == "Single Prediction":
    st.markdown("## ✨ Single Prediction")

    # Inputs
    col_dt1, col_dt2 = st.columns(2)
    with col_dt1:
        book_date = st.date_input("📅 Booking Date", value=pd.Timestamp.today().date())
    with col_dt2:
        book_time = st.time_input("⏰ Booking Time", value=DEFAULT_TIME)

    c1, c2 = st.columns(2)
    with c1:
        avg_vtat = st.number_input("📍 Avg VTAT", min_value=0.0, value=5.0, step=0.1,
                                   help="Vehicle travel/arrival time proxy")
    with c2:
        avg_ctat = st.number_input("🧭 Avg CTAT", min_value=0.0, value=8.0, step=0.1, help="Completion time proxy")

    c3, c4 = st.columns(2)
    with c3:
        vehicle_choice = st.selectbox("🚘 Vehicle Type", VEH_LABELS,
                                      index=VEH_LABELS.index("Bike") if "Bike" in VEH_LABELS else 0)
        pickup_choice = st.selectbox("📌 Pickup (top set)", PICK_LABELS,
                                     index=PICK_LABELS.index("Other") if "Other" in PICK_LABELS else 0)
    with c4:
        payment_choice = st.selectbox("💳 Payment Method", PAY_LABELS,
                                      index=PAY_LABELS.index("UPI") if "UPI" in PAY_LABELS else 0)
        drop_choice = st.selectbox("🎯 Drop (top set)", DROP_LABELS,
                                   index=DROP_LABELS.index("Other") if "Other" in DROP_LABELS else 0)

    # Build row
    row = {}
    for k, v in derive_time_features(book_date, book_time).items():
        if k in FEATURES_SET: row[k] = v
    if "Avg VTAT" in FEATURES_SET: row["Avg VTAT"] = float(avg_vtat)
    if "Avg CTAT" in FEATURES_SET: row["Avg CTAT"] = float(avg_ctat)
    set_onehot_group(row, VEH_LABELS, VEH_PREFIX, vehicle_choice)
    set_onehot_group(row, PAY_LABELS, PAY_PREFIX, payment_choice)
    set_onehot_group(row, PICK_LABELS, PICK_PREFIX, pickup_choice)
    set_onehot_group(row, DROP_LABELS, DROP_PREFIX, drop_choice)

    # What-if chart
    with st.expander("📈 What-if: how risk changes with Avg VTAT"):
        sim = []
        for v in np.linspace(max(0.0, avg_vtat - 20), avg_vtat + 20, 25):
            row_sim = row.copy();
            row_sim["Avg VTAT"] = float(v)
            Xsim = build_feature_row(row_sim)
            for c in ("hour", "weekday", "month"):
                if c in Xsim: Xsim[c] = Xsim[c].astype(np.int32)
            for c in ("Avg VTAT", "Avg CTAT"):
                if c in Xsim: Xsim[c] = Xsim[c].astype(np.float32)
            bool_cols = [c for c in Xsim.columns if
                         c.startswith(("vehicle_", "payment_", "pickup_", "drop_")) or c == "is_weekend"]
            if bool_cols: Xsim[bool_cols] = Xsim[bool_cols].astype(bool)
            p = float(model.predict_proba(Xsim)[:, 1][0]) if hasattr(model, "predict_proba") else 0.0
            sim.append({"Avg VTAT": v, "P(cancel)": p})
        df_sim = pd.DataFrame(sim)
        st.line_chart(df_sim.set_index("Avg VTAT"))

    st.divider()
    if st.button("🔮 Predict"):
        X1 = build_feature_row(row)

        # Cast to model-friendly dtypes
        for c in ("hour", "weekday", "month"):
            if c in X1.columns: X1[c] = X1[c].astype(np.int32)
        for c in ("Avg VTAT", "Avg CTAT"):
            if c in X1.columns: X1[c] = X1[c].astype(np.float32)
        bool_cols = [c for c in X1.columns if
                     c.startswith(("vehicle_", "payment_", "pickup_", "drop_")) or c == "is_weekend"]
        if bool_cols: X1[bool_cols] = X1[bool_cols].astype(bool)

        # Predict
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X1)[:, 1][0])
        else:
            score = float(model.decision_function(X1)[0])
            prob = 1.0 / (1.0 + np.exp(-score))
        pred = int(prob >= threshold)

        # -------- Visible, colorful output --------
        ratio = (prob / base_rate) if base_rate else None
        band = risk_band(prob, base_rate, threshold)
        band_class = "risk-high" if band == "High" else ("risk-med" if band == "Medium" else "risk-low")

        # Metrics with better spacing
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probability", f"{prob * 100:.1f}%")
        m2.metric("Baseline", f"{(base_rate or 0) * 100:.1f}%")
        m3.metric("× Baseline", f"{(ratio or 0):.1f}×")
        m4.metric("Decision", "Cancel" if pred == 1 else "Not Cancel")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"<span class='risk-chip {band_class}'>Risk band: {band}</span>", unsafe_allow_html=True)
        st.caption(f"Rule: predict **Cancel** when Probability ≥ Threshold ({threshold:.2f}).")

        # Cost-aware recommendation (flip card, enhanced spacing)
        cost_if_flag = expected_cost_if(True, prob, c_fp, c_fn, c_int)
        cost_if_notflag = expected_cost_if(False, prob, c_fp, c_fn, c_int)
        recommend_flag = cost_if_flag < cost_if_notflag
        rec_title = "FLAG (take action)" if recommend_flag else "DO NOT FLAG"
        hints = action_suggestions(band)
        flip_class = "flip-high" if band == "High" else ("flip-med" if band == "Medium" else "flip-low")

        # Enhanced Recommended Action Section - Simple Card Approach
        st.markdown("#### 🧭 Recommended action")

        # Create a simple, reliable card using Streamlit components
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
            <div style="background: {UBER_CARD}; padding: 20px; border-radius: 12px; border: 2px solid #2a2a2a; margin: 10px 0;">
                <h4 style="color: {UBER_TEXT}; margin: 0 0 10px 0;">{rec_title}</h4>
                <p style="color: {UBER_MUTED}; margin: 0;">Based on your input, here's what we recommend:</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background: {UBER_CARD}; padding: 20px; border-radius: 12px; border: 2px solid #2a2a2a; margin: 10px 0;">
                <h5 style="color: {UBER_YELLOW}; margin: 0 0 10px 0;">Cost Analysis</h5>
                <p style="color: {UBER_TEXT}; margin: 5px 0;">Act: {cost_if_flag:.2f}</p>
                <p style="color: {UBER_TEXT}; margin: 5px 0;">Don't: {cost_if_notflag:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        # Show suggested steps in an expander
        with st.expander("📋 View suggested steps", expanded=False):
            st.markdown("**Recommended actions:**")
            for i, hint in enumerate(hints, 1):
                st.markdown(f"{i}. {hint}")
            st.caption("Demo suggestions – company defines playbook in production.")

# ========================= Batch Prediction =========================
else:
    st.markdown("## 📦 Batch Prediction (CSV)")
    st.caption("CSV columns (optional helpers): date, time, Avg VTAT, Avg CTAT, vehicle, payment, pickup, drop")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df_raw = pd.read_csv(up)
        st.write("Preview:", df_raw.head())

        rows = []
        for _, r in df_raw.iterrows():
            d = {}
            # date/time
            try:
                dte = pd.to_datetime(r.get("date", pd.Timestamp.today())).date()
            except Exception:
                dte = pd.Timestamp.today().date()
            try:
                tme = pd.to_datetime(r.get("time", dtime(10, 0))).time()
            except Exception:
                tme = dtime(10, 0)
            for k, v in derive_time_features(dte, tme).items():
                if k in FEATURES_SET: d[k] = v
            # numerics
            if "Avg VTAT" in FEATURES_SET: d["Avg VTAT"] = float(r.get("Avg VTAT", np.nan))
            if "Avg CTAT" in FEATURES_SET: d["Avg CTAT"] = float(r.get("Avg CTAT", np.nan))
            # one-hots
            set_onehot_group(d, VEH_LABELS, VEH_PREFIX, str(r.get("vehicle", "")).strip())
            set_onehot_group(d, PAY_LABELS, PAY_PREFIX, str(r.get("payment", "")).strip())
            set_onehot_group(d, PICK_LABELS, PICK_PREFIX,
                             str(r.get("pickup", "")).strip() or ("Other" if "Other" in PICK_LABELS else ""))
            set_onehot_group(d, DROP_LABELS, DROP_PREFIX,
                             str(r.get("drop", "")).strip() or ("Other" if "Other" in DROP_LABELS else ""))
            rows.append(d)

        Xb = pd.DataFrame(rows, columns=FEATURES)
        for c in ("hour", "weekday", "month"):
            if c in Xb.columns: Xb[c] = Xb[c].astype(np.int32)
        for c in ("Avg VTAT", "Avg CTAT"):
            if c in Xb.columns: Xb[c] = Xb[c].astype(np.float32)
        bool_cols = [c for c in Xb.columns if
                     c.startswith(("vehicle_", "payment_", "pickup_", "drop_")) or c == "is_weekend"]
        if bool_cols: Xb[bool_cols] = Xb[bool_cols].astype(bool)

        if st.button("🚀 Run Batch Prediction"):
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(Xb)[:, 1]
            else:
                scores = model.decision_function(Xb)
                probs = 1.0 / (1.0 + np.exp(-scores))
            preds = (probs >= threshold).astype(int)

            out = df_raw.copy()
            out["cancel_prob"] = probs
            out["prediction"] = preds


            # Actionability in batch
            def _band_row(p):
                return risk_band(float(p), base_rate, threshold)


            def _action_row(b):
                return (action_suggestions(b) or ["No action"])[0]


            out["risk_band"] = out["cancel_prob"].map(_band_row)
            out["action"] = out["risk_band"].map(_action_row)

            # Cost-aware recommendation & expected saving
            c_fp = 1.0 if "c_fp" not in locals() else c_fp
            c_fn = 4.0 if "c_fn" not in locals() else c_fn
            c_int = 0.5 if "c_int" not in locals() else c_int
            out["recommend_flag"] = out["cancel_prob"].map(lambda p: expected_cost_if(True, p, c_fp, c_fn, c_int) <
                                                                     expected_cost_if(False, p, c_fp, c_fn, c_int))
            out["expected_cost_if_flag"] = out["cancel_prob"].map(
                lambda p: expected_cost_if(True, p, c_fp, c_fn, c_int))
            out["expected_cost_if_notflag"] = out["cancel_prob"].map(
                lambda p: expected_cost_if(False, p, c_fp, c_fn, c_int))
            out["expected_saving"] = out["expected_cost_if_notflag"] - out["expected_cost_if_flag"]

            st.success("Done. See results below.")
            st.write(f"Recommended to FLAG {int(out['recommend_flag'].sum())} / {len(out)} rides "
                     f"({out['recommend_flag'].mean():.1%}). "
                     f"Total expected saving: {out['expected_saving'].sum():.2f}")

            pretty = out.copy()
            pretty["cancel_prob"] = (pretty["cancel_prob"] * 100).round(1).astype(str) + "%"
            st.dataframe(pretty.head(100), use_container_width=True)

            st.download_button(
                "⬇️ Download results",
                data=out.to_csv(index=False),
                file_name="precancello_predictions.csv",
                mime="text/csv"
            )

# ----------------- Footer note (pushed lower) -----------------
st.markdown(
    "<p class='small-muted footer-note'>"
    "⚠️ Action Hints are demo suggestions for presentation. In production, the company defines the playbook and costs."
    "</p>",
    unsafe_allow_html=True
)
