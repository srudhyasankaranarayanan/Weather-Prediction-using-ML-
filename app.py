
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Weather Prediction System",
    page_icon="🌤️",
    layout="wide"
)

# --------------------------------
# 2. CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #F3E8FF 0%,
        #FCE7F3 35%,
        #FFF1E6 70%,
        #FEF3C7 100%
    );
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main heading */
h1 {
    color: #713B87 !important;
    text-align: center;
    font-size: 42px !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #8B4A9E !important;
    font-weight: 700 !important;
}

/* Paragraphs */
p, label {
    color: #51415F !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #E9D5FF,
        #FBCFE8,
        #FDE68A
    );
}

/* Input boxes */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTimeInput input {
    background-color: #FFFFFF !important;
    border: 2px solid #E9C5F5 !important;
    border-radius: 12px !important;
}

/* Select boxes */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E9C5F5;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    color: #A855F7;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(
        90deg,
        #A855F7,
        #EC4899,
        #F59E0B
    );
    color: white !important;
    border: none;
    border-radius: 14px;
    height: 50px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 5px 15px rgba(168, 85, 247, 0.3);
}

/* Form container */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.75);
    padding: 25px;
    border: 2px solid #E9D5FF;
    border-radius: 22px;
    box-shadow: 0 5px 20px rgba(120, 70, 150, 0.08);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #FFFFFF,
        #FCE7F3
    );
    padding: 22px;
    border-radius: 18px;
    border: 2px solid #F5C2E7;
    box-shadow: 0 5px 15px rgba(120, 70, 150, 0.10);
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #8B3FA8 !important;
    font-weight: 800;
}

/* Info messages */
[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 2px solid #E9D5FF;
    border-radius: 15px;
    overflow: hidden;
}

/* Horizontal divider */
hr {
    border-color: #D8B4FE;
}

/* Footer */
.footer {
    text-align: center;
    color: #80558F;
    font-size: 14px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# 3. LOAD TRAINED MODELS
# --------------------------------

@st.cache_resource
def load_models():

    temperature = joblib.load("temperature_model.pkl")
    precipitation = joblib.load("precipitation_model.pkl")
    weather = joblib.load("weather_model.pkl")

    return temperature, precipitation, weather


# --------------------------------
# 4. MAIN TITLE
# --------------------------------

st.title("🌤️ Weather Prediction System")

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "✨ Smart Weather Forecasting using Machine Learning ✨"
    "</p>",
    unsafe_allow_html=True
)

st.info("🌈 Forecast Horizon: Next One Hour")


# --------------------------------
# 5. CHECK MODEL FILES
# --------------------------------

model_files = [
    "temperature_model.pkl",
    "precipitation_model.pkl",
    "weather_model.pkl"
]

if not all(Path(file).exists() for file in model_files):

    st.error(
        "Trained model files are missing. "
        "Please run train_model.py first."
    )

    st.stop()

try:
    temperature_model, precipitation_model, weather_model = load_models()

except Exception as error:
    st.error(f"Unable to load models: {error}")
    st.stop()


# --------------------------------
# 6. USER INPUT SECTION
# --------------------------------

st.header("🌸 Enter Current Weather Details")

st.write(
    "Provide the current weather information "
    "to predict the upcoming weather."
)

with st.form("weather_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🌷 Atmospheric Details")

        humidity = st.slider(
            "Humidity (%)",
            0.0, 100.0, 70.0
        )

        apparent_temperature = st.number_input(
            "Apparent Temperature (°C)",
            value=28.0
        )

        wind_speed = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            value=10.0
        )

        wind_bearing = st.number_input(
            "Wind Bearing (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=180.0
        )

        visibility = st.number_input(
            "Visibility (km)",
            min_value=0.0,
            value=10.0
        )

    with col2:

        st.subheader("☀️ Weather Conditions")

        pressure = st.number_input(
            "Pressure (millibars)",
            min_value=0.0,
            value=1013.0
        )

        summary = st.selectbox(
            "Current Weather Summary",
            [
                "Clear",
                "Partly Cloudy",
                "Mostly Cloudy",
                "Overcast",
                "Foggy",
                "Breezy",
                "Breezy and Overcast",
                "Breezy and Mostly Cloudy",
                "Breezy and Partly Cloudy"
            ]
        )

        precipitation = st.selectbox(
            "Current Precipitation Type",
            ["rain", "snow"]
        )

        observation_time = st.time_input(
            "Observation Time",
            value=datetime.now().time().replace(
                second=0, microsecond=0
            )
        )

        observation_date = st.date_input(
            "Observation Date",
            value=datetime.now().date()
        )

    st.markdown("---")

    submit = st.form_submit_button(
        "✨ Predict Weather ✨"
    )


# --------------------------------
# 7. MAKE PREDICTIONS
# --------------------------------

if submit:

    observation_datetime = datetime.combine(
        observation_date,
        observation_time
    )

    input_data = pd.DataFrame([{
        "Humidity": humidity,
        "Apparent Temperature (C)": apparent_temperature,
        "Wind Speed (km/h)": wind_speed,
        "Wind Bearing (degrees)": wind_bearing,
        "Visibility (km)": visibility,
        "Pressure (millibars)": pressure,
        "Summary": summary,
        "Precip Type": precipitation,
        "hour": observation_datetime.hour,
        "month": observation_datetime.month,
        "day": observation_datetime.day,
        "day_of_week": observation_datetime.weekday()
    }])

    try:

        # Temperature prediction
        predicted_temperature = temperature_model.predict(
            input_data
        )[0]

        # Precipitation prediction
        predicted_precipitation = precipitation_model.predict(
            input_data
        )[0]

        # Weather condition prediction
        predicted_weather = weather_model.predict(
            input_data
        )[0]

        # --------------------------------
        # 8. DISPLAY RESULTS
        # --------------------------------

        st.markdown("---")

        st.header("🌈 Weather Prediction Results")

        forecast_time = observation_datetime + timedelta(hours=1)

        st.markdown(
            f"<p style='text-align:center; color:#8B4A9E; "
            f"font-size:18px; font-weight:bold;'>"
            f"📅 Forecast for: "
            f"{forecast_time.strftime('%d-%m-%Y %I:%M %p')}"
            f"</p>",
            unsafe_allow_html=True
        )

        result1, result2, result3 = st.columns(3)

        with result1:
            st.metric(
                "🌡️ Temperature",
                f"{predicted_temperature:.2f} °C"
            )

        with result2:
            st.metric(
                "🌧️ Precipitation",
                str(predicted_precipitation).title()
            )

        with result3:
            st.metric(
                "☁️ Weather Condition",
                str(predicted_weather)
            )

        st.success(
            "🎉 Weather prediction completed successfully!"
        )

        # Input details
        st.subheader("📋 Input Weather Details")

        st.dataframe(
            input_data,
            use_container_width=True
        )

    except Exception as error:

        st.error(f"Prediction failed: {error}")


# --------------------------------
# 9. FOOTER
# --------------------------------

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        🌸 Weather Prediction System 🌸<br>
        Developed using Python, Pandas, Scikit-learn and Streamlit<br>
        ✨ Machine Learning Based Weather Forecasting ✨
    </div>
    """,
    unsafe_allow_html=True
)
