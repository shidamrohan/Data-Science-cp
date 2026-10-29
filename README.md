# Smart Retail Analytics
**Machine Learning-Based Sales Forecasting, Customer Segmentation and Product Demand Prediction**

An end-to-end interactive Data Science and Predictive Analytics platform built with Python, Streamlit, Plotly, Scikit-Learn, and XGBoost on the Sample Superstore retail dataset. All financial metrics and predictions are calibrated in **Indian Rupees (₹)** at current exchange rates.

---

## 📌 Business Questions Answered

| Module | Core Business Question | Business Impact |
|--------|------------------------|-----------------|
| **Executive Dashboard** | How much revenue and profit does the business generate? | High-level performance KPIs & monthly velocity |
| **Sales Analytics** | Which products, categories, and regions perform best? | Identifies regional growth & the *Discount Trap* |
| **Sales Forecast** | What revenue will the business generate next month? | Multi-model regression with 6-month projections |
| **Demand Forecast** | How many units will each product/sub-category require? | Reorder recommendations & stockout risk prevention |
| **Customer Segmentation** | Who are the most valuable customers? | RFM + K-Means ($K=4$) persona-based targeting |
| **Product Analytics** | What actions should the business take on products? | Star vs Loss-leader detection with automated actions |
| **Model Performance** | Which algorithm predicts sales with highest accuracy? | Benchmark of Linear Regression vs Random Forest vs XGBoost |

---

## 🚀 Quickstart & Setup

### 1. Clone the repository
```bash
git clone https://github.com/shidamrohan/Data-Science-cp.git
cd Data-Science-cp
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Using standard venv
python -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate    # On Linux/macOS

pip install -r requirements.txt
```

### 3. Run the Machine Learning Pipeline (One-shot)
```bash
python src/train_all.py
```
This script:
- Cleans and preprocesses the raw Superstore transaction records.
- Converts currency to INR (`1 USD = ₹95.28`).
- Engineers temporal lag features, rolling statistics, and cyclical seasonality.
- Builds and normalizes RFM features.
- Trains and saves all models: Linear Regression, Random Forest, XGBoost, and K-Means.

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🧠 Real Model Performance (Held-out Test Split)

| Model | MAE (₹) | RMSE (₹) | R² Score | Status |
|-------|---------|----------|----------|--------|
| **Random Forest Regressor** | **₹84,668** | **₹1,23,733** | **0.6122** | 🏆 **Best Model** |
| XGBoost Regressor | ₹92,203 | ₹1,31,774 | 0.5602 | Solid baseline |
| Linear Regression | ₹1,00,319 | ₹1,35,288 | 0.5364 | Linear baseline |

- **Product Demand Model (RF)**: $R^2 = \mathbf{0.7104}$, $\text{MAE} = 14.80\text{ units}$
- **Customer Segmentation (K-Means $K=4$)**: Silhouette Score = $\mathbf{0.2695}$ (Segments: *VIP Customers, Regular Customers, Occasional Shoppers, At-Risk Customers*)

---

## 🛠️ Technology Stack

| Layer | Tools |
|-------|-------|
| **Frontend Dashboard** | Streamlit, Plotly Express & Graph Objects |
| **Machine Learning** | Scikit-Learn, XGBoost, SciPy |
| **Data Processing** | Pandas, NumPy, PyArrow |
| **Model Persistence** | Joblib |
| **Testing** | Pytest |

---

## 📂 Project Architecture

```
smart-retail-analytics/
├── app.py                         # Executive Overview Dashboard (Home)
├── requirements.txt               # Pinned production dependencies
├── README.md                      # Documentation & project guide
├── .gitignore                     # Git ignore rules
│
├── data/
│   ├── raw/
│   │   └── SampleSuperstore.csv   # Historical retail transactions
│   └── processed/                 # Generated Parquet datasets
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Data cleansing & INR currency pipeline
│   ├── feature_engineering.py     # Lags, rolling averages, RFM builder
│   ├── formatters.py              # Full Indian number formatting (₹ Lakhs/Crores)
│   ├── train_sales.py             # Sales revenue regression (LR, RF, XGB)
│   ├── train_demand.py            # Unit volume demand forecasting
│   ├── clustering.py              # RFM normalization + K-Means (K=4)
│   ├── evaluation.py              # Metrics loader (MAE, RMSE, R², Silhouette)
│   └── train_all.py               # Complete end-to-end pipeline runner
│
├── pages/
│   ├── 1_📊_Sales_Analytics.py      # Granular filters, trends & discount trap
│   ├── 2_📈_Sales_Forecast.py       # Live revenue forecast with growth predictor
│   ├── 3_📦_Demand_Forecast.py      # Inventory risk & recommended stock levels
│   ├── 4_👥_Customer_Segmentation.py # 2D/3D RFM cluster visualization & targeting
│   ├── 5_🏷️_Product_Analytics.py     # Margin rankings, loss leaders & recommendations
│   └── 6_🧠_Model_Performance.py    # Academic viva & model benchmarking page
│
├── models/                        # Pre-trained model artifacts (.pkl, .csv)
└── tests/
    └── test_pipeline.py           # Unit tests
```
