import joblib
import pandas as pd

# Load model
model = joblib.load("models/xgb_rain_model.pkl")

# Example new input
new_data = pd.DataFrame({
    'PRCP':[2.3],
    'TAVG':[15.2],
    'Year':[2019],
    'Month':[5],
    'Day':[10],
    'DayOfYear':[130],
    'WeekOfYear':[19],
    'PRCP_Lag1':[1.2],
    'PRCP_Lag2':[0.0],
    'TAVG_Lag1':[14.8],
    'TAVG_Lag2':[13.5],
    'PRCP_3day_avg':[1.5],
    'PRCP_7day_avg':[1.1],
    'Month_sin':[0.5],
    'Month_cos':[0.8],
    'DayOfYear_sin':[0.2],
    'DayOfYear_cos':[0.9]
})

prediction = model.predict(new_data)

print("Rain Tomorrow:" , prediction[0])
