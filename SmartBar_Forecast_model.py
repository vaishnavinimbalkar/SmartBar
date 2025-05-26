import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
import math

# Load data
df = pd.read_csv("Consumption Dataset - Dataset.csv")

# Preprocessing
df['Date Time Served'] = pd.to_datetime(df['Date Time Served'])
df = df.rename(columns={
    'Consumed (ml)': 'consumed',
    'Bar Name': 'bar',
    'Brand Name': 'brand',
    'Alcohol Type': 'alcohol'
})

# Filter out rows where opening + purchase is zero implying there was no stock to sell
df = df[(df['Opening Balance (ml)'] + df['Purchase (ml)']) > 0]

# Create 'week' column as the start of the week
df['week'] = df['Date Time Served'].dt.to_period("W").apply(lambda r: r.start_time)

# Aggregate weekly consumption
weekly_df = (
    df.groupby(['bar', 'alcohol', 'brand', 'week'])['consumed']
    .sum()
    .reset_index(name='weekly_consumed_ml')
)

# Encode categorical variables
label_encoders = {}
for col in ['bar', 'alcohol', 'brand']:
    le = LabelEncoder()
    weekly_df[col + '_enc'] = le.fit_transform(weekly_df[col])
    label_encoders[col] = le

# Feature Engineering
weekly_df = weekly_df.sort_values(['bar', 'brand', 'week'])
weekly_df['week_number'] = weekly_df.groupby(['bar', 'brand']).cumcount()
weekly_df['month'] = weekly_df['week'].dt.month
weekly_df['is_peak'] = weekly_df['month'].isin([1, 5, 12]).astype(int)  
weekly_df['lag_1'] = weekly_df.groupby(['bar', 'brand'])['weekly_consumed_ml'].shift(1)
weekly_df['lag_2'] = weekly_df.groupby(['bar', 'brand'])['weekly_consumed_ml'].shift(2)
weekly_df = weekly_df.dropna(subset=['lag_1', 'lag_2'])

# Demand type label
def demand_label(val):
    if val <= 200:
        return 0  # Low
    elif val <= 450:
        return 1  # Medium
    return 2  # High

weekly_df['demand_type'] = weekly_df['weekly_consumed_ml'].apply(demand_label)

# Feature columns
features = ['week_number', 'bar_enc', 'brand_enc', 'alcohol_enc', 'is_peak', 'lag_1', 'lag_2']

# Round to nearest 50ml
def round_up_50(x):
    return int(math.ceil(x / 50.0) * 50)

# Forecast loop
forecast_results = []

groups = weekly_df.groupby(['bar', 'alcohol', 'brand'])

for (bar, alcohol, brand), group in groups:
    group = group.sort_values('week_number').reset_index(drop=True)
    
    if len(group) < 6:
        continue
    
    train = group.iloc[:-1]
    test = group.iloc[-1:]
    
    # Train regression model
    reg_model = XGBRegressor(n_estimators=50, learning_rate=0.1)
    reg_model.fit(train[features], train['weekly_consumed_ml'])
    forecast = reg_model.predict(test[features])[0]
    forecast = max(0, forecast)

    # Train classifier for demand type
    cls_model = XGBClassifier(n_estimators=50)
    cls_model.fit(train[features], train['demand_type'])
    demand_pred = cls_model.predict(test[features])[0]

    # Safety stock and par level
    safety_stock_factor = {0: 0.10, 1: 0.20, 2: 0.30}[demand_pred]
    safety_stock = forecast * safety_stock_factor
    par_level = forecast + safety_stock

    forecast_results.append({
        'Bar': bar,
        'Alcohol Type': alcohol,
        'Brand': brand,
        'Forecasted 7-Day Consumption (ml)': round(forecast, 2),
        'Predicted Demand Type': ['Low', 'Medium', 'High'][demand_pred],
        'Safety Stock (ml)': round(safety_stock, 2),
        'Recommended Par Level (ml)': round(par_level, 2),
        'Recommended Par Level Rounded (ml)': round_up_50(par_level)
    })

# Save results
forecast_df = pd.DataFrame(forecast_results)
forecast_df.to_csv("7_day_forecast_per_bar_brand.csv", index=False)
print(forecast_df.head())
