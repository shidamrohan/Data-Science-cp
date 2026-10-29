# 📘 Project Document: Smart Retail Sales Forecasting & Customer Analytics

---

## 1. Executive Summary
**Project Title:** Smart Retail Analytics: Machine Learning-Based Sales Forecasting, Customer Segmentation, and Product Demand Prediction  
**Domain:** Retail Analytics, Predictive Modeling, Machine Learning, Business Intelligence  
**Dataset:** Sample Superstore Retail Transaction Dataset (~9,994 records across 21 business attributes)  
**Currency:** Indian Rupees (₹) calibrated at exchange rate `1 USD = ₹95.28`  
**Tech Stack:** Python 3.11, Streamlit, Plotly, Scikit-Learn, XGBoost, Pandas, NumPy, Joblib  

---

## 2. What is this Project? (WHAT)
This project is an **end-to-end Data Science and Decision-Support System** designed for modern retail chains and supermarkets. 

Unlike traditional academic projects that build isolated, disconnected machine learning scripts, this application unifies **four core data science pillars into a single interactive commercial dashboard**:

```
                              ┌──────────────────────────────────────────────┐
                              │     Sample Superstore Retail Transactions    │
                              │      (Dates, Customers, Products, Profit)    │
                              └──────────────────────┬───────────────────────┘
                                                     │
                                       [ Data Preprocessing Pipeline ]
                                       - Currency Normalization (₹ INR)
                                       - Deduplication & Outlier Clipping
                                                     │
                   ┌─────────────────────────────────┼────────────────────────────────┐
                   ▼                                 ▼                                ▼
       [ 1. Sales Forecasting ]           [ 2. Demand Forecasting ]       [ 3. Customer Segmentation ]
       - Linear Regression Baseline       - Monthly Unit Quantity Model   - RFM Matrix Extraction
       - Random Forest Regressor          - Safety Stock Buffer (+15%)    - Log-Scale & Normalization
       - XGBoost Regressor                - Shortage Risk (High/Med/Low)  - K-Means Clustering (K=4)
       - Evaluated via MAE, RMSE, R²                                      - 4 Commercial Personas
                   │                                 │                                │
                   └─────────────────────────────────┼────────────────────────────────┘
                                                     ▼
                                     [ 4. Interactive Web Dashboard ]
                                  Streamlit + Plotly 2D/3D Visualizations
```

1. **Sales Revenue Forecasting:** Predicts expected storewide and regional sales revenue (₹) for future periods using historical trends, seasonality, and promotional discount levels.
2. **Product Demand & Inventory Intelligence:** Forecasts unit demand volume per product/sub-category and recommends optimal stock levels with automated shortage risk detection.
3. **RFM Customer Segmentation:** Analyzes transaction history using Recency, Frequency, and Monetary metrics to segment customers into actionable marketing groups via K-Means clustering.
4. **Product Analytics & Decision Support:** Uncovers the "Discount Trap" (products with high sales volume but negative profit margins) and provides automated business recommendations.

---

## 3. Why Was It Built? (WHY)

### The Business Problems in Retail
1. **The Retrospective Reporting Trap:** Most retail businesses only look at retrospective BI dashboards (what already happened in the past). They lack predictive intelligence to anticipate what *will* happen next month.
2. **Stockouts vs. Overstocking:** 
   - Stockouts (running out of hot items) cause lost revenue and frustrated shoppers.
   - Overstocking ties up working capital in warehouse storage and risks product obsolescence.
3. **The "Discount Trap":** Retailers often offer heavy discounts (20%–40%) to boost top-line revenue, unaware that it is eroding profit margins and causing negative cash flow in specific sub-categories like Tables or Bookcases.
4. **Blanket Marketing Inefficiency:** Treating all customers the same wastes marketing budget. Retailers need to know who their VIPs are (who need loyalty perks) versus At-Risk customers (who need reactivation discounts).

---

## 4. How Does It Work? (HOW)

### Step 1: Data Cleaning & Preprocessing (`src/preprocessing.py`)
- Reads raw CSV files with multi-encoding fallbacks (`utf-8`, `latin-1`, `cp1252`).
- Parses dates and removes invalid records and duplicate orders.
- Applies standard interquartile range (IQR × 3) outlier clipping.
- Converts monetary values into Indian Rupees (₹) at `1 USD = ₹95.28`.
- Derives helper dimensions: `Year`, `Month`, `Quarter`, `YearMonthStr`, `Profit_Margin %`, and `Revenue_Per_Unit`.

### Step 2: Feature Engineering (`src/feature_engineering.py`)
- **Temporal Lags:** Computes $t-1, t-2, t-3$ previous sales and unit quantity lags.
- **Rolling Averages:** Generates 3-month and 6-month moving averages (`Sales_Roll3`, `Sales_Roll6`) to capture momentum.
- **Cyclical Encoding:** Converts calendar months into continuous trigonometric coordinates ($\sin(2\pi m / 12)$ and $\cos(2\pi m / 12)$) to prevent December-January boundary discontinuity.
- **One-Hot Encoding:** Encodes categorical variables (`Category`, `Region`, `Sub-Category`).

### Step 3: Sales Forecasting Pipeline (`src/train_sales.py`)
- Splits monthly aggregated data into an **80/20 time-ordered train/test split** (preserving chronological order to prevent data leakage).
- Trains and benchmarks three distinct regression models:
  1. **Linear Regression** (Standard baseline)
  2. **Random Forest Regressor** (Ensemble of 200 trees)
  3. **XGBoost Regressor** (Gradient boosted trees)
- Evaluates real metrics: **MAE**, **RMSE**, and **$R^2$ Score**.

### Step 4: Product Demand & Inventory Modeling (`src/train_demand.py`)
- Aggregates monthly unit quantity per Sub-Category.
- Trains a Random Forest demand model to forecast future units.
- Calculates **Recommended Reorder Stock**:
  $$\text{Recommended Stock} = \text{Predicted Demand} \times (1 + \text{Safety Buffer})$$
  *(Default buffer = 15%)*
- Categorizes **Shortage Risk**:
  - 🔴 **HIGH RISK:** Predicted demand exceeds current stock baseline by $>25\%$.
  - 🟡 **MEDIUM RISK:** Within $\pm 10\%$.
  - 🟢 **LOW RISK:** Stock buffer is safe.
- Automatically assigns products into **High**, **Medium**, or **Low Demand Tiers** based on historical velocity percentiles.

### Step 5: RFM Customer Segmentation (`src/clustering.py`)
- Calculates three behavioral metrics for every unique customer:
  - **Recency ($R$):** Days elapsed since the customer's last order relative to the dataset snapshot date (`max(Order Date) + 1 day`).
  - **Frequency ($F$):** Total number of distinct orders placed.
  - **Monetary ($M$):** Total lifetime rupee spend by the customer.
- Applies `log1p` transformation to reduce positive skewness inherent in retail spend distributions.
- Normalizes data via `StandardScaler` and applies **K-Means Clustering ($K=4$)**.
- Automatically ranks cluster centroids and assigns human-interpretable retail personas:
  - 👑 **VIP Customers:** High spend, frequent orders, recent visits $\rightarrow$ *Action: Loyalty perks, early access.*
  - ⭐ **Regular Customers:** Steady purchases, consistent value $\rightarrow$ *Action: Personalized cross-selling.*
  - 🎟️ **Occasional Shoppers:** Infrequent orders, seasonal $\rightarrow$ *Action: Targeted promotional discounts.*
  - ⚠️ **At-Risk Customers:** High historical spend, but long lapse since last purchase $\rightarrow$ *Action: Automated win-back campaigns.*

### Step 6: Interactive Dashboard (`app.py` & `pages/`)
- Built with **Streamlit** and interactive **Plotly** graphics.
- Features custom CSS ensuring full Indian Rupee values (`₹1,57,82,368`) display without truncation or `...` ellipsis.
- Fast page loading achieved by loading pre-trained `.pkl` models via `joblib`.

---

## 5. Summary of Real Results

### A. Sales Forecasting Model Benchmark (Real Test Set)
| Model | MAE (₹) | RMSE (₹) | $R^2$ Score | Outcome |
|---|---|---|---|---|
| **Random Forest Regressor** | **₹84,668** | **₹1,23,733** | **0.6122** | 🏆 **Winner (Selected Model)** |
| XGBoost Regressor | ₹92,203 | ₹1,31,774 | 0.5602 | Strong non-linear model |
| Linear Regression | ₹1,00,319 | ₹1,35,288 | 0.5364 | Baseline comparison |

*Top Predictive Features:* Monthly order count (43.2%), Quantity sold (14.1%), 3-Month Rolling Average (9.6%), Seasonal month cosine (4.5%), Discount rate (4.3%).

### B. Product Demand Model
- **Algorithm:** Random Forest Regressor
- **Validation Accuracy:** $R^2 = \mathbf{0.7104}$, $\text{MAE} = 14.80\text{ units}$

### C. Customer Segmentation
- **Algorithm:** K-Means ($K=4$) on normalized log-RFM features
- **Validation:** Silhouette Score = $\mathbf{0.2695}$ across 793 unique retail customers.

---

## 6. Project Navigation & Modules

| Module | Filename | Description |
|---|---|---|
| **Executive Dashboard** | `app.py` | Storewide summary: Total Sales, Total Profit, Orders, Customers, AOV, monthly trends, and regional breakdown. |
| **Sales Analytics** | `pages/1_📊_Sales_Analytics.py` | Multi-filter deep dive into dates, regions, sub-categories, YoY comparisons, and the Discount Trap scatter plot. |
| **Sales Forecast** | `pages/2_📈_Sales_Forecast.py` | Interactive revenue simulator: select Category, Region, and Discount to predict next month's sales with 6-month projections. |
| **Demand Forecast** | `pages/3_📦_Demand_Forecast.py` | Unit volume forecasting: select any Product or Sub-Category, view expected demand, recommended stock, and stockout risk badge. |
| **Customer Segmentation** | `pages/4_👥_Customer_Segmentation.py` | RFM cluster explorer: 2D & 3D rotating scatter plots, customer breakdown, and marketing targeting matrix. |
| **Product Analytics** | `pages/5_🏷️_Product_Analytics.py` | Product rankings: Top/bottom 10 by revenue and profit, loss leaders list, and automated decision-support advisories. |
| **Model Performance** | `pages/6_🧠_Model_Performance.py` | Academic evaluation: side-by-side model comparison table, feature importance chart, and full pipeline diagram. |

---

## 7. How to Run & Verify

```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Run training pipeline (regenerates all models & metrics in ~10 seconds)
python src/train_all.py

# 3. Launch interactive web dashboard
streamlit run app.py
```
App opens at: **`http://localhost:8501`**

---

## 8. Viva & Interview Questions Cheat Sheet

- **Q: Why did you choose Random Forest over Linear Regression?**  
  *A:* Retail sales data exhibits non-linear relationships and interactions (e.g., discounts improve sales volume up to a threshold, after which margins collapse). Random Forest captures these interactions without assuming linear independence, resulting in a higher $R^2$ (0.6122 vs 0.5364) and lower MAE.

- **Q: Why did you log-transform RFM features before K-Means?**  
  *A:* Customer transaction frequency and monetary spending follow a power-law (Pareto) distribution with severe right skewness. K-Means assumes spherical clusters with equal variance. Applying $\log(1 + x)$ compresses the long tail and normalizes feature distributions, significantly improving cluster quality.

- **Q: How does demand forecasting differ from sales forecasting?**  
  *A:* Sales forecasting predicts dollar revenue to assist financial planning and cash-flow management. Demand forecasting predicts physical unit counts to prevent warehouse stockouts and calculate safety stock reorder buffers.

