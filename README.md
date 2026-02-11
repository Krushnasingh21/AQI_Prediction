# # 🌫️ Air Quality Index (AQI) Prediction

A machine learning project to predict the Air Quality Index (AQI) of Indian cities using historical pollutant data. Three regression models are trained and compared — Linear Regression, Random Forest, and XGBoost — with hyperparameter tuning applied to the best-performing model.

---

## 📁 Repository Structure

```
aqi-prediction/
│
├── data/
│   ├── city_day.csv              # Raw dataset (29,531 daily records across Indian cities)
│   └── aqi_predictions.csv       # Model-generated AQI predictions on the test set
│
├── notebooks/
│   └── AQI.ipynb                 # Main Jupyter notebook (EDA, preprocessing, modeling)
│
└── README.md
```

---

## 📊 Dataset

**File:** `data/city_day.csv`

Daily air quality readings collected from monitoring stations across major Indian cities, spanning **2015 to 2020**.

| Column | Description |
|---|---|
| `City` | Name of the Indian city |
| `Date` | Date of the reading |
| `PM2.5`, `PM10` | Particulate matter concentrations |
| `NO`, `NO2`, `NOx`, `NH3` | Nitrogen compounds |
| `CO`, `SO2`, `O3` | Carbon monoxide, Sulfur dioxide, Ozone |
| `Benzene`, `Toluene`, `Xylene` | Volatile organic compounds |
| `AQI` | Air Quality Index (target variable) |
| `AQI_Bucket` | AQI category (Good / Satisfactory / Moderate / Poor / Very Poor / Severe) |

> **Note:** The dataset contains significant missing values in several columns (e.g., Xylene: ~61%, PM10: ~38%), which are handled during preprocessing.

---

## 🔬 Methodology

The notebook `AQI.ipynb` walks through the full pipeline:

1. **Data Loading & Exploration** — Overview of shape, data types, and null value distribution
2. **Exploratory Data Analysis (EDA)** — Visualizations of pollutant distributions and correlations
3. **Preprocessing** — Null value handling, feature selection, and dropping non-numeric columns
4. **Train-Test Split** — 80% training / 20% testing with `random_state=42`
5. **Feature Scaling** — StandardScaler applied to normalize input features
6. **Model Training & Evaluation** — Three models compared using MAE, RMSE, and R² Score
7. **Hyperparameter Tuning** — GridSearchCV with 5-fold cross-validation applied to XGBoost
8. **Prediction Export** — Final predictions saved to `data/aqi_predictions.csv`

---

## 📈 Model Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 31.20 | 59.12 | 0.81 |
| XGBoost | 21.91 | 42.90 | 0.90 |
| **Random Forest** | **20.84** | **40.87** | **0.91** |

> **Random Forest** achieved the best overall performance. XGBoost was further tuned via GridSearchCV over `n_estimators`, `learning_rate`, and `max_depth`.

---

## ⚙️ Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/aqi-prediction.git
   cd aqi-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook notebooks/AQI.ipynb
   ```

---

## 📌 Notes

- The notebook was originally developed on Google Colab. If running locally, update the data path in the notebook from `/content/city_day (1).csv` to `../data/city_day.csv`.
- The predictions file (`aqi_predictions.csv`) contains the predicted AQI values for the test split using the tuned XGBoost model.
