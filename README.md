# 🌦️ Heathrow Weather Analytics & Machine Learning Project

## 📌 Project Overview

The project presents a complete **End-to-End Data Analysis and Machine Learning Pipeline** using Heathrow weather data (2010-2019).
This project focuses on building a machine learning model to predict rainfall (precipitation) at Heathrow meteorological data from 2010 to 2019. The dataset includes daily measurements of precipitation (PRCP), average temperature (TAVG), and other attributes. The model uses XGBoost for regression-based prediction of rainfall amounts.

**Key components:**
- Data Source: Historical weather data from Heathrow (station: UKM00003772).
- Exploratory Data Analysis (EDA): Performed in Heathrow_EDA.ipynb to understand trends, correlations, and visualizations (e.g., temperature trends by month and rainfall status).
- Model Training: Implemented in model.ipynb using libraries like Pandas, NumPy, Scikit-learn, XGBoost, and others. The trained model is saved as Heathrow_rain_model.pkl.
- Prediction Focus: Predicts daily precipitation (PRCP) using features like average temperature (TAVG), date-based features (e.g., month, year), and derived indicators (e.g., raining or not).

The project demonstrates data preprocessing, feature engineering, model training, evaluation, and visualization of feature importance.


# 📂 Project Structure

```
Heathrow data analysis/
│
├── data/
│   ├── HeathrowMeteo2010-2019.csv 
│   ├── HeathrowMeteo2010-2019.xlsx   
│   └── Processed_df.csv                   
│
├── notebooks/
│   ├── Heathrow_EDA.ipynb                 
│   └── model.ipynb                     
│
├── models/
│   └── Heathrow_rain_model.pkl           
│
├── dashboard/
│   └── Heathrow_dashboard.pbix
│
├── src/                                
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── outputs/
│   ├── dashboard.png
│   ├── feature_importance.png
│   └── monthly_temperature_distribution_(EDA).png
│   
│
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE                     

```

# 📊 Dataset Description

- File: HeathrowMeteo2010-2019.csv
- Source: Likely from NOAA or similar meteorological database.
- Columns:
    - STATION: Station ID (UKM00003772).
    - NAME: Location (HEATHROW, UK).
    - LATITUDE, LONGITUDE, ELEVATION: Geographical details.
    - DATE: Date of measurement (YYYY-MM-DD).
    - PRCP: Precipitation in mm (target variable).
    - PRCP_ATTRIBUTES: Data flags.
    - TAVG: Average temperature in °C.
    - TAVG_ATTRIBUTES: Data flags.

- Size: 3621 entries (daily data from 2010-01-01 to 2019-12-31).
- Missing Values: ~30 in PRCP (handled by median imputation).
- Processed Data: Generated in EDA notebook as Processed_df.csv with additional features like Year, Month, Day, and Raining (binary).

### Key Engineered Features:

-   Rainfall Binning
-   Seasonal Categorization
-   Rolling Averages
-   Time-based Aggregations
-   PRCP_Lag1, PRCP_Lag2
-   TAVG_Lag1, TAVG_Lag2
-   PRCP_3day_avg
-   PRCP_7day_avg
-   Cyclical Encoding:
    -   Month_sin / Month_cos
    -   DayOfYear_sin / DayOfYear_cos
- In dashboard:
    -   Temperature Anomaly
    -   Rain Intensity Categories
    -   Rain Flag (Binary)
    -   Rolling 12-Month Average
    -   Year-over-Year (YoY) Change
    -   Month-over-Month (MoM) Change


# 🔎 Exploratory Data Analysis (EDA)

Performed in `Heathrow_EDA.ipynb`:

### ✔ Data Cleaning

-   Removed duplicates
-   Handled missing values
-   Dropped unnecessary columns

### ✔ Feature Engineering

-   Rainfall Binning
-   Seasonal Categorization
-   Rolling Averages
-   Time-based Aggregations

### ✔ Visual Analysis

-   Temperature Trends by Year
-   Rainfall Distribution
-   Correlation Analysis (Temp vs Rain)
-   Seasonal Patterns
-   Time-Series Decomposition

### 📌 Key Insights from EDA

-   Clear seasonal temperature patterns
-   Rainfall distribution heavily skewed toward low precipitation days
-   Moderate correlation between temperature and rainfall
-   Increasing variability in extreme rainfall events
-   Data cleaning and handling missing values (e.g., filling PRCP with median).
-   Time-based feature extraction (e.g., year, month, day from DATE).
-   Binary classification for raining days (PRCP > 0).
-   Regression model for predicting rainfall amount.
-   Visualizations: Heatmaps for monthly average temperatures on rainy/dry days, feature importance bar charts.
-   Model evaluation metrics (not explicitly shown in snippets but implied via training/test splits).
-   Saved model for inference.


# 🤖 Machine Learning Model

Implemented in `model.ipynb`

### Models Used:

-   Linear Regression
-   Random Forest
-   XGBoost
-   LightGBM
-   Sequential model (TensorFlow / Keras , Neural Network)
-   Prophet (Time-Series Forecasting)

### Workflow:

1.  Train-Test Split
2.  Feature Scaling
3.  Model Training
4.  Hyperparameter Tuning
5.  Model Evaluation

### Evaluation Metrics:

-   Accuracy score
-   MAE
-   MSE
-   RMSE
-   R² Score
-   Classification report

### Model Export:

Final trained model saved as:

-   model/Heathrow_rain_model.pkl


# 📈 Power BI Dashboard

Interactive dashboard built in Power BI featuring:

-   Average Temperature KPI
-   Average Teperature MOM KPI
-   Rainy Days Percentage KPI
-   Precipitation YoY Change KPI
-   Monthly Temperature Trends
-   Rainfall Intensity Distribution
-   Rolling 12M Average Forecast
-   Temperature and precipitation relation per month and year

### Dashboard Insights:

-   Seasonal temperature peaks in July-August
-   High rainfall concentration in specific years
-   Visible YoY precipitation fluctuations
-   Long-term temperature trend stabilization


# 🛠 Installation

### 1️⃣ Clone Repository

    git clone https://github.com/Hemantdhake/heathrow-data-analysis-and-rainfall-prediction.git
    cd heathrow-data-analysis-and-rainfall-prediction

### 2️⃣ Create Virtual Environment (Recommended)

    python -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate    # Windows

### 3️⃣ Install Dependencies

-   `pip install -r requirements.txt`


#  Usage

### Exploratory Data Analysis Notebook

    Run Heathrow_EDA.ipynb to perform EDA:

        - Load and preprocess data.
        - Generate visualizations (e.g., heatmaps for temperature trends).
        - Create Processed_df.csv for modeling.

    jupyter notebook Heathrow_EDA.ipynb

### Model Training Notebook

    Run model.ipynb to train the model:

        - Loads data from CSV.
        - Feature engineering (e.g., date components, lagging features implied).
        - Trains XGBoost regressor.
        - Evaluates feature importance.
        - Saves the model as Heathrow_rain_model.pkl.

    jupyter notebook notebooks/model.ipynb

### Open Power BI Dashboard

Open:

    dashboard/Heathrow_Weather_Dashboard.pbix


# 📊 Results

-   Temperature Trends: Heatmaps show higher average temperatures in summer months (June-August), with rainy days slightly cooler than dry days.
-   Rainfall Patterns: More rainy days in winter (e.g., November-February). Annual rainy days visualized via bar plots.
-   Correlations: PRCP and TAVG may have weak correlations; explored via pairplots.
-   Outliers: Handled implicitly via median imputation.
-   Ensemble models (XGBoost / LightGBM) outperformed linear regression.
-   Prophet model effectively captured seasonality.
-   Dashboard enables interactive climate trend exploration.


# 🚀 Skills Demonstrated

-   Data Cleaning & Preprocessing
-   Feature Engineering
-   Time-Series Analysis
-   Supervised Learning
-   Model Evaluation & Optimization
-   Data Visualization (Matplotlib / Power BI)
-   Business Insight Communication

# 🔮 Future Work

1. Advanced Time-Series Modeling

-   Current approach:
        - Supervised learning with lag features
-   Future improvements:
        - Use LSTM / GRU for sequence modeling
        - Try Temporal Fusion Transformer (TFT)
        - Implement Prophet for seasonal modeling
        - Use TimeSeriesSplit cross-validation
This can better capture long-term seasonal dependencies.

2. Hyperparameter Optimization

-   Instead of manual tuning:
        - GridSearchCV
        - RandomizedSearchCV
        - Optuna (Bayesian Optimization)


# 🤝 Contribution Guidelines

Contributions are welcome! Please fork the repository and submit a pull request with improvements

1.  Fork the repository
2.  Create a new branch
3.  Commit your changes
4.  Push to your branch
5.  Open a Pull Request


# 📬 Contact

-   Email : hemant.kjwaa22@sinhgad.edu
-   🔗[LinkedIn](https://www.linkedin.com/in/hemant-dhake-4606a8301/)
