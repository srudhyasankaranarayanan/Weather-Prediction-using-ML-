# Weather Prediction System using Machine Learning

## Project Overview

The Weather Prediction System is a machine learning project that predicts upcoming weather conditions using historical weather data.

This project uses machine learning algorithms to predict the next-hour temperature, precipitation type, and weather condition based on the current weather details.

The application is developed using Python and Streamlit with a colourful and user-friendly dashboard.

## Features

- Predict next-hour temperature.
- Predict precipitation type (Rain or Snow).
- Predict weather conditions.
- Enter current weather details manually.
- Display prediction results using metric cards.
- Interactive and colourful Streamlit dashboard.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

## Machine Learning Algorithms

This project uses:

1. Random Forest Regressor – To predict temperature.
2. Random Forest Classifier – To predict precipitation type.
3. Random Forest Classifier – To predict weather condition.

## Dataset

The project uses historical weather data containing the following attributes:

- Formatted Date
- Summary
- Precip Type
- Temperature (C)
- Apparent Temperature (C)
- Humidity
- Wind Speed (km/h)
- Wind Bearing (degrees)
- Visibility (km)
- Pressure (millibars)
- Daily Summary

## Project Structure

```text
Weather-Prediction-ML/
│
├── weather.csv
├── weather_cleaned.csv
├── inspect_data.py
├── clean_data.py
├── eda.py
├── train_model.py
├── evaluate_model.py
├── app.py
│
├── temperature_model.pkl
├── precipitation_model.pkl
├── weather_model.pkl
│
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/srudhyasankaranarayanan/Weather-Prediction-using-ML
```

### 2. Navigate to the Project Folder

```bash
cd Weather-Prediction-ML
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
```

### 6. Train the Machine Learning Models

```bash
python train_model.py
```

This generates three trained model files:

- temperature_model.pkl
- precipitation_model.pkl
- weather_model.pkl

### 7. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## How It Works

1. The historical weather dataset is loaded.
2. The data is cleaned and explored.
3. Important weather features are selected.
4. Machine learning models are trained using the dataset.
5. The trained models are evaluated.
6. The user enters current weather details in the Streamlit application.
7. The system predicts the next-hour temperature, precipitation type, and weather condition.
8. The results are displayed on the dashboard.

## Expected Output

The application displays:

- Predicted Temperature in Celsius.
- Predicted Precipitation Type.
- Predicted Weather Condition.
- Forecast date and time.
- User-entered weather details.

## Future Enhancements

- Add real-time weather API integration.
- Include more machine learning algorithms.
- Improve prediction accuracy.
- Add weather trend visualizations.
- Deploy the application online.

## Conclusion

The Weather Prediction System demonstrates how machine learning can be used to analyse historical weather data and predict upcoming weather conditions. The project combines data preprocessing, machine learning, model evaluation, and Streamlit to create an interactive weather forecasting application.

## Author

**Srudhya Sankaranarayanan**

**Project:** Weather Prediction System using Machine Learning
