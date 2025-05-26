# SmartBar 🍸  
**ML-Driven Inventory Forecasting and Optimization System for Hotel Chain Bars**

## 📌 Overview

SmartBar is a machine learning-powered system designed to help hotel-chain bars optimize their inventory through weekly consumption forecasting and data-driven par level recommendations. The system reduces stockouts and overstocking by predicting demand per brand, alcohol type, and bar location.

This repository contains the core forecasting script that:
- Cleans and aggregates alcohol consumption data,
- Predicts 7-day consumption per bar-brand using XGBoost,
- Classifies demand as Low, Medium, or High,
- Calculates safety stock and recommended par levels.

## 🎯 Problem Statement

Bars in hotel chains often:
- Run out of fast-moving stock (leading to lost sales),
- Over-purchase slow-moving brands (causing waste),
- Lack consistent restocking strategy across locations.

**SmartBar** addresses these problems with:
- Automated, weekly demand forecasting per item,
- Demand-type-based safety stock estimation,
- Par level recommendations for smarter inventory planning.

---

## 🧠 Core Assumptions

1. Customers are brand-loyal (no brand switching).
2. Each bar operates independently; no stock sharing.
3. All bars stock their inventory once a week
4. Input data is manually recorded and infrequent.
5. Weekly inventory replenishment replaces ad-hoc ordering.

---

## 🔍 Dataset & Input

Expected CSV structure:

| Date Time Served | Bar Name | Brand Name | Alcohol Type | Opening Balance (ml) | Purchase (ml) | Consumed (ml) |
|------------------|----------|------------|--------------|----------------------|---------------|---------------|

Data should be saved as: `Consumption Dataset - Dataset.csv`.

---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Converts timestamps and renames columns.
   - Filters out rows with zero stock availability.
   - Aggregates consumption data weekly.

2. **Feature Engineering**
   - Encodes `bar`, `brand`, and `alcohol type`.
   - Adds lag features (previous 2 weeks' consumption).
   - Flags seasonal peaks (Jan, May, Dec).

3. **ML Models**
   - `XGBRegressor`: Forecasts next week's consumption.
   - `XGBClassifier`: Predicts demand type (Low, Medium, High).

4. **Par Level Recommendation**
   - Safety stock is calculated based on demand type:
     - Low: +10%
     - Medium: +20%
     - High: +30%
   - Par level = forecast + safety stock, rounded to the nearest 50ml.

5. **Output**
   - Results saved to `7_day_forecast_per_bar_brand.csv`.

---

## 📈 Sample Output

| Bar            | Alcohol Type | Brand      | Forecasted 7-Day Consumption (ml) | Predicted Demand Type | Safety Stock (ml) | Recommended Par Level (ml) | Recommended Par Level Rounded (ml) |
|----------------|--------------|------------|-----------------------------------|------------------------|--------------------|-----------------------------|------------------------------------|
| Taylor’s Bar   | Beer         | Budweiser  | 390.0                             | Medium                 | 78.0               | 468.0                       | 500                                |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/smartbar-forecast.git
cd smartbar-forecast
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost
```

### 3. Add your dataset

Place `Consumption Dataset - Dataset.csv` in the root directory.

### 4. Run the forecast script

```bash
python forecast.py
```

### 5. View the output

Check the generated `7_day_forecast_per_bar_brand.csv`.

---

## 📌 Future Enhancements

- Integrate with hotel POS systems and inventory databases.
- Incorporate external data: weather, events, promotions.
- Transition from weekly to daily forecasting.
- Optimize across multiple bars with stock-swapping suggestions.
- Deploy using a cloud-based dashboard for real-time visibility.

---

## 🧑‍💻 Author

**Vaishnavi Nimbalkar**  
B. Tech Electronics & Computer Engineering  
Vellore Institute of Technology

---

## 📬 Contact

For any queries, feel free to reach out or fork the project!
