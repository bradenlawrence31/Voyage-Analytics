"""
Voyage Analytics - Streamlit Dashboard
=======================================
Run with: streamlit run streamlit_app.py

Make sure your Flask API is running first:
    python api/app.py
"""

import requests
import pandas as pd
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:5000"

CITIES = [
    "Aracaju (SE)", "Brasilia (DF)", "Campo Grande (MS)",
    "Florianopolis (SC)", "Natal (RN)", "Recife (PE)",
    "Rio de Janeiro (RJ)", "Salvador (BH)", "Sao Paulo (SP)"
]
AGENCIES   = ["FlyingDrops", "CloudFy", "Rainbow"]
FLIGHT_TYPES = ["economic", "premium", "firstClass"]
COMPANIES  = ["4You", "Acme Factory", "Monsters CYA", "Umbrella LTDA", "Wonka Company"]

# ── Page Setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0 0.25rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1E3A5F, #2563EB);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    .result-label {
        font-size: 0.85rem;
        font-weight: 400;
        opacity: 0.8;
        margin-bottom: 0.25rem;
    }
    .hotel-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .hotel-rank {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2563EB;
    }
    .hotel-name {
        font-size: 1rem;
        font-weight: 600;
        color: #1E3A5F;
    }
    .api-status-ok {
        background: #DCFCE7;
        color: #166534;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .api-status-err {
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .section-divider {
        border: none;
        border-top: 2px solid #E2E8F0;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper ──────────────────────────────────────────────────────────────────────
def call_api(endpoint, payload):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to Flask API. Make sure `python api/app.py` is running."
    except Exception as e:
        return None, str(e)


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        return data.get("models", {})
    except:
        return None


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">✈️ Voyage Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MLOps Capstone Project · Travel Intelligence Dashboard</div>', unsafe_allow_html=True)

# API Status Bar
health = check_api_health()
col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns([2, 1, 1, 1, 2])
with col_s2:
    if health:
        status = "🟢 Flight Model" if health.get("flight_price_model") else "🔴 Flight Model"
        st.markdown(f'<span class="api-status-ok">{status}</span>', unsafe_allow_html=True)
with col_s3:
    if health:
        status = "🟢 Gender Model" if health.get("gender_classifier") else "🔴 Gender Model"
        st.markdown(f'<span class="api-status-ok">{status}</span>', unsafe_allow_html=True)
with col_s4:
    if health:
        status = "🟢 Hotel Model" if health.get("hotel_recommender") else "🔴 Hotel Model"
        st.markdown(f'<span class="api-status-ok">{status}</span>', unsafe_allow_html=True)

if not health:
    st.error("⚠️ Flask API is not reachable. Please run `python api/app.py` first.")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "✈️  Flight Price Predictor",
    "👤  Gender Classifier",
    "🏨  Hotel Recommender",
    "📊  Dataset Insights"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FLIGHT PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("✈️ Flight Price Predictor")
    st.caption("Enter flight details below to get a predicted ticket price.")

    col1, col2 = st.columns(2)

    with col1:
        origin      = st.selectbox("Departure City", CITIES, key="f_from")
        flight_type = st.selectbox("Flight Class", FLIGHT_TYPES, key="f_type")
        agency      = st.selectbox("Booking Agency", AGENCIES, key="f_agency")

    with col2:
        destination = st.selectbox("Destination City", CITIES, index=1, key="f_to")
        distance    = st.number_input("Distance (km)", min_value=100.0, max_value=5000.0, value=676.53, step=10.0)
        time        = st.number_input("Flight Duration (hours)", min_value=0.5, max_value=24.0, value=3.5, step=0.5)

    if origin == destination:
        st.warning("⚠️ Departure and destination cities cannot be the same.")
    else:
        if st.button("🔍 Predict Flight Price", use_container_width=True, key="btn_flight"):
            payload = {
                "from": origin, "to": destination,
                "flightType": flight_type, "time": time,
                "distance": distance, "agency": agency
            }
            with st.spinner("Predicting..."):
                result, error = call_api("/predict/flight", payload)

            if error:
                st.error(error)
            else:
                price = result.get("predicted_flight_price", "N/A")
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">Predicted Flight Price</div>
                    R$ {price:,.2f}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📋 Trip Summary")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                summary_col1.metric("Route", f"{origin} → {destination}")
                summary_col2.metric("Class", flight_type.capitalize())
                summary_col3.metric("Agency", agency)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GENDER CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("👤 Gender Classifier")
    st.caption("Predict a user's gender based on their profile information.")

    col1, col2 = st.columns(2)

    with col1:
        g_name    = st.text_input("Full Name", value="Roy Braun")
        g_company = st.selectbox("Company", COMPANIES)

    with col2:
        g_age = st.slider("Age", min_value=18, max_value=70, value=30)

    if st.button("🔍 Classify Gender", use_container_width=True, key="btn_gender"):
        payload = {"company": g_company, "name": g_name, "age": g_age}
        with st.spinner("Classifying..."):
            result, error = call_api("/predict/gender", payload)

        if error:
            st.error(error)
        else:
            predicted = result.get("predicted_gender", "N/A").capitalize()
            probs     = result.get("probabilities", {})

            gender_emoji = "👨" if predicted.lower() == "male" else "👩" if predicted.lower() == "female" else "🧑"
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Predicted Gender</div>
                {gender_emoji} {predicted}
            </div>
            """, unsafe_allow_html=True)

            if probs:
                st.markdown("#### 📊 Prediction Confidence")
                prob_cols = st.columns(len(probs))
                for i, (label, score) in enumerate(probs.items()):
                    prob_cols[i].metric(label.capitalize(), f"{score*100:.1f}%")
                    prob_cols[i].progress(float(score))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HOTEL RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🏨 Hotel Recommender")
    st.caption("Get personalized hotel recommendations based on your travel preferences.")

    col1, col2 = st.columns(2)

    with col1:
        h_place   = st.selectbox("Destination", CITIES, key="h_place")
        h_days    = st.number_input("Number of Days", min_value=1, max_value=30, value=4)

    with col2:
        h_budget  = st.number_input("Max Budget per Day (R$)", min_value=50.0, max_value=500.0, value=300.0, step=10.0)
        h_top_n   = st.slider("Number of Recommendations", min_value=1, max_value=10, value=3)

    if st.button("🔍 Find Hotels", use_container_width=True, key="btn_hotel"):
        payload = {
            "place": h_place, "days": h_days,
            "max_price": h_budget, "top_n": h_top_n
        }
        with st.spinner("Finding best hotels..."):
            result, error = call_api("/recommend/hotels", payload)

        if error:
            st.error(error)
        else:
            recs = result.get("recommendations", [])
            if not recs:
                st.warning("No hotels found for your criteria.")
            else:
                st.markdown(f"#### 🏆 Top {len(recs)} Hotels for {h_place}")
                for i, hotel in enumerate(recs, 1):
                    in_budget = hotel['price_per_day'] <= h_budget
                    budget_tag = "✅ Within Budget" if in_budget else "⚠️ Over Budget"
                    total = hotel.get("estimated_total", hotel['price_per_day'] * h_days)

                    st.markdown(f"""
                    <div class="hotel-card">
                        <span class="hotel-rank">#{i}</span>
                        <span class="hotel-name"> &nbsp; {hotel['hotel_name']}</span>
                        &nbsp;&nbsp; <small style="color:#6B7280">{hotel['place']}</small>
                        &nbsp;&nbsp; <small style="color:{'#166534' if in_budget else '#991B1B'}">{budget_tag}</small>
                        <br><br>
                        <b>R$ {hotel['price_per_day']:.2f}</b> / day &nbsp;·&nbsp;
                        <b>R$ {total:.2f}</b> total for {h_days} days &nbsp;·&nbsp;
                        {hotel['booking_count']:,} bookings &nbsp;·&nbsp;
                        Score: {hotel['score']:.2f}
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATASET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Dataset Insights")
    st.caption("Visual exploration of the Voyage Analytics datasets.")

    # Load data
    @st.cache_data
    def load_data():
        flights = pd.read_csv("datasets/flights.csv")
        hotels  = pd.read_csv("datasets/hotels.csv")
        users   = pd.read_csv("datasets/users.csv")
        return flights, hotels, users

    try:
        flights_df, hotels_df, users_df = load_data()

        # ── KPI Row ──────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("✈️ Total Flights",   f"{len(flights_df):,}")
        k2.metric("🏨 Total Hotels",    f"{hotels_df['name'].nunique():,}")
        k3.metric("👤 Total Users",     f"{len(users_df):,}")
        k4.metric("🌆 Cities Covered",  f"{flights_df['from'].nunique():,}")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Row 1: Flight charts ──────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✈️ Flights by Class**")
            flight_type_counts = flights_df["flightType"].value_counts().reset_index()
            flight_type_counts.columns = ["Flight Class", "Count"]
            st.bar_chart(flight_type_counts.set_index("Flight Class"))

        with col2:
            st.markdown("**🏢 Flights by Agency**")
            agency_counts = flights_df["agency"].value_counts().reset_index()
            agency_counts.columns = ["Agency", "Count"]
            st.bar_chart(agency_counts.set_index("Agency"))

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Row 2: Price distributions ────────────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**💰 Average Flight Price by City (Departure)**")
            avg_price = flights_df.groupby("from")["price"].mean().sort_values(ascending=False).reset_index()
            avg_price.columns = ["City", "Avg Price (R$)"]
            st.bar_chart(avg_price.set_index("City"))

        with col4:
            st.markdown("**🏨 Average Hotel Price by City**")
            avg_hotel = hotels_df.groupby("place")["price"].mean().sort_values(ascending=False).reset_index()
            avg_hotel.columns = ["City", "Avg Price/Day (R$)"]
            st.bar_chart(avg_hotel.set_index("City"))

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Row 3: User insights ──────────────────────────────────────────────
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("**👥 Gender Distribution**")
            gender_counts = users_df["gender"].value_counts().reset_index()
            gender_counts.columns = ["Gender", "Count"]
            st.bar_chart(gender_counts.set_index("Gender"))

        with col6:
            st.markdown("**🏢 Users by Company**")
            company_counts = users_df["company"].value_counts().reset_index()
            company_counts.columns = ["Company", "Count"]
            st.bar_chart(company_counts.set_index("Company"))

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Raw data preview ──────────────────────────────────────────────────
        st.markdown("**🗂️ Raw Data Preview**")
        dataset_choice = st.selectbox("Select Dataset", ["Flights", "Hotels", "Users"])
        if dataset_choice == "Flights":
            st.dataframe(flights_df.head(20), use_container_width=True)
        elif dataset_choice == "Hotels":
            st.dataframe(hotels_df.head(20), use_container_width=True)
        else:
            st.dataframe(users_df.head(20), use_container_width=True)

    except FileNotFoundError:
        st.error("⚠️ Dataset files not found. Make sure the `datasets/` folder is in your project root.")


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#9CA3AF; font-size:0.8rem;'>"
    "Voyage Analytics · MLOps Capstone Project · Built with Streamlit & Flask"
    "</p>",
    unsafe_allow_html=True
)