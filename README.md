# AI-Based Retail Customer Analytics, Data Warehouse and Churn Prediction

Dataset: https://www.kaggle.com/datasets/carrie1/ecommerce-data (UK online retailer, Dec 2010 - Dec 2011)

## Project layout
```
config.py           Set the path to data.csv here (one place for everything)
preprocessing.py    Cleaning report: what was wrong, what was removed, what was replaced
app.py              Streamlit dashboard (6 sections)
notebook.py         Cell-by-cell notebook for VS Code (export to .ipynb for submission)
src/pipeline.py     All logic: cleaning, star schema, OLAP, cube, churn model
data/data.csv       Put the Kaggle file here
```

## Run in VS Code
1. Download `data.csv` from Kaggle into `data/`.
2. Open the folder in VS Code, then in the terminal:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   python preprocessing.py     # cleaning report -> outputs/ (cleaned_data.csv, cleaning_report.txt, cleaning_summary.png)
   streamlit run app.py        # dashboard
   ```
3. For the notebook: open `notebook.py`, pick the `.venv` interpreter, click **Run Cell** above each `# %%`.
   Export with the `...` menu -> *Export* -> `.ipynb`.

## Deploy on Streamlit Community Cloud
1. Push the folder (including `data/data.csv`, about 45 MB, under GitHub's 100 MB limit) to a GitHub repo.
2. Go to share.streamlit.io -> New app -> choose the repo, branch, main file `app.py`.
3. Deploy. First load runs the pipeline for about a minute, then it is cached.

## Requirement -> where it is implemented
| # | Requirement | Where |
|---|-------------|-------|
| 1 | Collection & understanding | `load_raw`, `quality_report` |
| 2-3 | Cleaning, missing values, outliers | `clean` |
| 4 | Integration & transformation | `transform` |
| 5 | Reduction & discretization | `reduce`, `discretize`, SpendBand |
| 6 | Quality & consistency | `consistency_checks` |
| 7 | Star schema | `build_star_schema`, `save_to_sqlite` |
| 8 | OLAP | `roll_up`, `drill_down`, `slice_cube`, `dice_cube`, `pivot` |
| 9 | Data cube | `data_cube` |
| 10-11 | Churn model & evaluation | `build_churn_dataset`, `train_and_evaluate` |
| 12 | Insights | `business_insights` |
