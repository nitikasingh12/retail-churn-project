# %% [markdown]
# # AI-Based Retail Customer Analytics, Data Warehouse and Churn Prediction
# Run cell by cell in VS Code (Python Interactive window). Export to `.ipynb` via
# the `...` menu → *Export* for submission, or upload the `.ipynb` to Kaggle.
#
# On Kaggle: add the dataset `carrie1/ecommerce-data` and upload `src/pipeline.py`
# (as a utility script or dataset) so `import pipeline` works.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from src import pipeline as P
except ImportError:           # Kaggle: pipeline.py uploaded next to the notebook
    sys.path.append("/kaggle/working")
    import pipeline as P

pd.set_option("display.float_format", "{:,.2f}".format)
KAGGLE = Path("/kaggle/input/ecommerce-data/data.csv")
from config import DATA_PATH
DATA = KAGGLE if KAGGLE.exists() else Path(DATA_PATH)

# %% [markdown]
# ## 1. Data collection and understanding

# %%
raw = P.load_raw(DATA)
print(raw.shape)
display(raw.head())
display(raw.dtypes.to_frame("dtype"))
display(raw.describe())

# %% [markdown]
# ## 2-3. Cleaning, missing values, duplicates, outliers

# %%
display(P.quality_report(raw))
raw[["Quantity", "UnitPrice"]].clip(-100, 100).plot.box(subplots=True, figsize=(8, 3), title="Raw (clipped to ±100 for display)")
plt.show()

# %%
df, cancels, log, caps = P.clean(raw)
display(log)
display(caps)
df[["Quantity", "UnitPrice"]].plot.box(subplots=True, figsize=(8, 3), title="After outlier capping")
plt.show()

# %% [markdown]
# ## 4-5. Integration, transformation, reduction, discretization

# %%
df = P.discretize(P.transform(df))
df, memory = P.reduce(df)
print("Memory (MB):", memory)
display(df.PriceBand.value_counts(sort=False).to_frame("rows"))
display(df.QtyBand.value_counts(sort=False).to_frame("rows"))

# %% [markdown]
# ## 6. Quality and consistency checks

# %%
display(P.consistency_checks(df))

# %% [markdown]
# ## 7. Data warehouse – star schema (fact_sales + 4 dimensions)

# %%
tables = P.build_star_schema(df)
for name, t in tables.items():
    print(f"{name}: {t.shape}")
    display(t.head(3))
P.save_to_sqlite(tables, "warehouse.db")           # real SQL warehouse file
base = P.olap_base(tables)

# %% [markdown]
# ## 8. OLAP operations

# %%
# Roll-up (Year) -> Drill-down (Year, Quarter) -> Drill-down (Year, Quarter, Month)
H = P.HIERARCHIES["Time"]
display(P.roll_up(base, H, 1))
display(P.drill_down(base, H, 1))
monthly = P.roll_up(base, H, 3)
monthly.plot(x="YearMonth", y="TotalPrice", kind="bar", figsize=(10, 3), title="Revenue by month")
plt.show()

# %%
# Slice: Region = UK
display(P.slice_cube(base, "Region", "UK").TotalPrice.sum())
# Dice: 2011, non-UK regions, Top/High spenders
dice = P.dice_cube(base, {"Year": [2011], "Region": ["Europe", "Asia-Pacific", "North America"], "SpendBand": ["Top", "High"]})
display(dice.groupby(["Region", "SpendBand"], observed=True).TotalPrice.sum().unstack())
# Pivot
pt = P.pivot(base, "Region", "Quarter")
display(pt)
pt.plot(kind="bar", stacked=True, figsize=(8, 3), title="Revenue: Region x Quarter")
plt.show()

# %% [markdown]
# ## 9. Data cube

# %%
cube = P.data_cube(base, ["Year", "Region", "SpendBand"])
print(len(cube), "cells in", 2 ** 3, "cuboids")
display(cube.sort_values("Revenue", ascending=False).head(15))

# %% [markdown]
# ## 10. Churn prediction

# %%
data, cutoff = P.build_churn_dataset(df, cancels)
print("Cut-off:", cutoff, "| churn rate:", round(data.Churn.mean(), 3))
display(data.head())
ml = P.train_and_evaluate(data)
display(ml["results"])

# %% [markdown]
# ## 11. Model evaluation

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for n, (fpr, tpr) in ml["roc"].items():
    ax[0].plot(fpr, tpr, label=n)
ax[0].plot([0, 1], [0, 1], "k--")
ax[0].set(title="ROC curve", xlabel="FPR", ylabel="TPR")
ax[0].legend()
cm = ml["cms"][ml["best"]]
ax[1].imshow(cm, cmap="Blues")
ax[1].set(title=f"Confusion matrix – {ml['best']}", xticks=[0, 1], yticks=[0, 1],
          xticklabels=["Active", "Churn"], yticklabels=["Active", "Churn"])
for i in range(2):
    for j in range(2):
        ax[1].text(j, i, cm[i, j], ha="center", va="center")
ml["importance"].sort_values().plot.barh(ax=ax[2], title="Feature importance")
plt.tight_layout()
plt.show()

# %%
scores = P.score_customers(df, cancels, ml["models"][ml["best"]], ml["columns"])
display(scores[scores.Risk == "High"].head(15))

# %% [markdown]
# ## 12. Business insights and conclusion

# %%
for line in P.business_insights(base, data, ml, scores):
    print("-", line)
