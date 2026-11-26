# 🛒 AI-Based Retail Customer Analytics, Data Warehouse & Churn Prediction

An end-to-end data analytics and data warehousing project built using real-world retail transaction data from a UK-based online retailer. This project transforms raw transactional data into a structured analytical data warehouse, performs OLAP operations, and applies Machine Learning to predict customer churn.

---

## 📌 Project Overview

This project focuses on transforming raw e-commerce transactional data into meaningful business insights using data preprocessing, data integration, dimensional modeling, OLAP analysis, data cube operations, and Machine Learning.

The system processes retail transaction records, builds a Star Schema-based data warehouse, enables multidimensional analysis, and predicts customer churn using historical customer purchasing behaviour.

## 🎯 Objectives

- Understand and analyze real-world retail transactional data.
- Perform data cleaning, missing value handling, outlier treatment and data transformation.
- Implement data reduction and discretization techniques.
- Design and implement a Star Schema data warehouse.
- Perform OLAP operations for multidimensional analysis.
- Generate and analyze a Data Cube.
- Develop a Machine Learning model for customer churn prediction.
- Visualize business insights through an interactive Streamlit dashboard.

## 📊 Dataset

**Dataset:** [Online Retail Dataset – Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

The dataset contains transactional records from a UK-based online retailer covering December 2010 to December 2011.

### Dataset attributes

| Column | Description |
|---|---|
| InvoiceNo | Unique invoice identifier |
| StockCode | Product identifier |
| Description | Product name or description |
| Quantity | Number of products purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per product |
| CustomerID | Unique customer identifier |
| Country | Customer's country |

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Data processing and ML implementation |
| Pandas & NumPy | Data cleaning and transformation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning and evaluation |
| SQLite | Data warehouse storage |
| Streamlit | Interactive dashboard |
| Jupyter Notebook | Experimentation and academic presentation |

## 🏗️ Project Architecture

```text
Raw Dataset
     │
     ▼
Data Collection & Understanding
     │
     ▼
Data Cleaning & Quality Assessment
     │
     ▼
Data Integration & Transformation
     │
     ▼
Data Reduction & Discretization
     │
     ▼
Data Warehouse (Star Schema)
     │
     ▼
OLAP Operations & Data Cube
     │
     ▼
Customer Analytics & Churn Prediction
     │
     ▼
Interactive Streamlit Dashboard
```

## 📁 Project Structure

```text
retail-customer-analytics/
│
├── app.py
├── config.py
├── preprocessing.py
├── notebook.py
├── requirements.txt
│
├── src/
│   └── pipeline.py
│
├── data/
│   └── data.csv
│
├── outputs/
│   ├── cleaned_data.csv
│   ├── cleaning_report.txt
│   └── cleaning_summary.png
│
├── database/
│   └── retail_dw.sqlite
│
└── README.md
```

## 🔄 Project Workflow

### 1. Data Collection & Understanding

- Load the raw retail dataset.
- Analyze dataset dimensions, data types and distributions.
- Identify missing values, duplicates and inconsistent records.
- Generate a data quality report.

### 2. Data Cleaning

- Handle missing customer identifiers.
- Remove duplicate records.
- Handle invalid transaction records.
- Identify and treat anomalies.
- Validate quantity and pricing data.
- Generate a cleaning summary.

### 3. Data Integration & Transformation

- Convert transaction dates into appropriate datetime formats.
- Calculate transaction revenue.
- Extract year, month, day and other date-related attributes.
- Aggregate transactional data into customer-level metrics.
- Generate transformed analytical features.

### 4. Data Reduction & Discretization

- Reduce the dataset through aggregation and feature selection.
- Create customer-level analytical datasets.
- Discretize continuous variables.
- Generate categorical spending groups such as `SpendBand`.

### 5. Data Quality & Consistency

- Validate data types.
- Check invalid numerical values.
- Validate relationships between fact and dimension tables.
- Verify consistency of transaction-level and aggregated metrics.

### 6. Data Warehouse Design

Implement a Star Schema with a central fact table and supporting dimension tables.

**Fact Table:**

- FactSales

**Dimension Tables:**

- DimCustomer
- DimProduct
- DimDate
- DimCountry

The schema enables efficient analytical queries across customers, products, dates and geographical locations.

### 7. OLAP Operations

Implement multidimensional data analysis:

- **Roll-up:** Aggregate sales data across higher-level dimensions.
- **Drill-down:** Explore data at more granular levels.
- **Slice:** Analyze a specific subset of the data.
- **Dice:** Filter data using multiple dimensions.
- **Pivot:** Rotate analytical views to compare dimensions.

### 8. Data Cube

Generate a multidimensional Data Cube to analyze retail sales across different dimensions, including:

- Time
- Customers
- Products
- Countries

The Data Cube supports aggregated analysis and helps identify patterns in sales performance and customer behaviour.

### 9. Customer Churn Prediction

Develop a Machine Learning-based customer churn prediction system using customer purchasing behaviour and derived features.

Potential customer features include:

- Recency
- Frequency
- Monetary Value
- Average Order Value
- Purchase Frequency
- Customer Lifetime Activity

Churn is defined using a selected inactivity period and a reference date. The churn definition and feature cutoff should be documented to avoid data leakage.

The model supports customer risk identification and customer retention analysis.

## 📈 Dashboard Features

The Streamlit dashboard provides an interactive interface for:

- Dataset overview and statistics.
- Data quality and cleaning analysis.
- Sales and customer analytics.
- Data Warehouse schema exploration.
- OLAP operations.
- Data Cube visualization.
- Customer churn prediction and model analysis.

## 🚀 Installation & Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/retail-customer-analytics.git
cd retail-customer-analytics
```

### Step 2: Download the dataset

Download the dataset from Kaggle and place it inside:

```text
data/data.csv
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the preprocessing pipeline

```bash
python preprocessing.py
```

This generates the cleaned dataset, cleaning report and summary visualization inside the `outputs/` directory.

### Step 6: Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📓 Notebook

The project includes `notebook.py`, a cell-based Python notebook compatible with VS Code.

To execute:

1. Open `notebook.py` in VS Code.
2. Select the Python interpreter associated with `.venv`.
3. Execute each cell using the Run Cell option.
4. Export the notebook to `.ipynb` if required for academic submission.

## 🌐 Deployment

The application can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Select `app.py` as the main application file.
5. Deploy the application.

For production deployment, use a hosted dataset or external storage instead of committing large datasets directly to GitHub.

## 📚 Data Mining & Data Warehousing Concepts Implemented

| Concept | Implementation |
|---|---|
| Data Collection | Loading and understanding raw data |
| Data Cleaning | Missing values, duplicates, anomalies |
| Data Integration | Combining and transforming features |
| Data Transformation | Feature engineering and aggregation |
| Data Reduction | Customer-level aggregation and feature reduction |
| Discretization | SpendBand and categorical feature generation |
| Data Quality | Consistency checks and validation |
| Data Warehouse | Star Schema |
| OLAP | Roll-up, Drill-down, Slice, Dice, Pivot |
| Data Cube | Multidimensional aggregation |
| Machine Learning | Customer churn classification |
| Visualization | Interactive Streamlit dashboard |

## 🔮 Future Enhancements

- Compare multiple churn classification algorithms.
- Integrate customer segmentation using clustering.
- Add interactive sales forecasting.
- Implement automated ETL workflows.
- Add model explainability using SHAP.
- Improve dashboard deployment and scalability.
- Add downloadable analytical reports.

## 👩‍💻 Author

**Nitika Singh**

B.Tech Computer Science Engineering (AI & ML)

---

*An end-to-end Data Mining, Data Warehousing and Machine Learning project designed to transform retail transaction data into actionable business insights.*
