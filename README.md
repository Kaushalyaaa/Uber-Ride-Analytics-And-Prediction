# 🚖 User-Focused Data Mining System for Ride-Hailing

A data-mining and ML pipeline that turns raw ride-booking data into **actionable insights** for riders:
- 📈 predict surge/peak windows,
- 💳 estimate monthly spend,
- 🗺️ discover frequent routes & rider segments,
- 💡 recommend money-saving subscription plans via an interactive dashboard.

---

## 📜 1) Project Background

Ride-hailing platforms complete millions of trips daily, but **predicting demand, pricing, cancellations, and satisfaction** remains challenging.  
This project uses a Kaggle dataset of ~148k Uber rides (India/NCR) with fields like pickup/drop-off, time, vehicle, fare, rating, payment, and booking status to mine patterns and build predictive models that improve decision-making for riders and ops teams.

---

## 🗂️ 2) Dataset (Brief Introduction)

-**Source**: Kaggle — *ncr_ride_bookings.csv*  [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard/data?select=ncr_ride_bookings.csv)

- **Key fields**: booking_id, timestamps, pickup, drop-off, fare, distance, payment method, rating, booking/cancellation status.  
- **Why it’s suitable**: rich temporal + geospatial signals enable demand surges, route patterns, and spend forecasts.  

> Engineered features used in modeling include:
> - `num_rides` — rides per time window per pickup area.
> - `high_demand` — binary flag for surge/peak periods.

---

## 🎯 3) Business Goals

1. 💰 **Help riders save money**: compare predicted monthly spend vs. subscription plan options.  
2. ⏱️ **Avoid peak pain**: warn users about busy-time windows and expected surge.  
3. 🚦 **Explain behavior**: show frequent routes & rider segments (commuters vs. leisure).  
4. 🖥️ **Make it usable**: deliver everything through a simple, responsive **Streamlit dashboard**.  

---

## 🧠 4) What’s Implemented (Data Mining & ML)

### 🔹 Data Preparation
- Handling missing values, duplicates, inconsistent formats.  
- Clean, analysis-ready dataset saved to `data/processed/`.

### 🔹 Feature Engineering
- `num_rides` (rides per pickup × time-slot), `high_demand` surge flag, ride duration, average ratings by location.  

### 🔹 Pattern Discovery
- Apriori / FP-Growth frequent route mining.  
- Association analysis for time slots × demand × booking preferences.  
- Clustering (K-Means, DBSCAN) for rider segmentation.  

### 🔹 Predictive Modeling
- Regression → forecast monthly spend.  
- Classification → predict high-demand windows.  
- Recommendation → subscription savings.  

### 🔹 Evaluation
- Metrics: RMSE, Accuracy/F1, Silhouette.  
- Emphasis on interpretability & real savings.  

### 🔹 Visualization & UI
- Heatmaps, savings comparison charts, cluster persona views.  
- Streamlit dashboard with alerts & insights.  

---

## 🛠️ 5) Tools & Technologies

- 🐍 **Python** (pandas, numpy, scikit-learn)  
- 🔍 **Mining**: mlxtend (Apriori), efficient-apriori  
- 🧩 **Clustering**: scikit-learn (KMeans, DBSCAN)  
- 📊 **Visualization**: matplotlib, seaborn, plotly  
- 🖥️ **Dashboard**: Streamlit  
- 🗂️ **Versioning**: git  
- ✅ **Quality**: pytest  
- ☁️ **Google Colab** 
