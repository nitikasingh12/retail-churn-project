"""Retail analytics pipeline: cleaning -> star schema -> OLAP -> data cube -> churn model."""
from __future__ import annotations

import io
import sqlite3
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTLIER_Q = 0.995          # winsorising percentile
CHURN_HORIZON_DAYS = 90    # churn = no purchase in the last 90 days of the data
NON_EU = {"Australia": "Asia-Pacific", "Japan": "Asia-Pacific", "Singapore": "Asia-Pacific",
          "Hong Kong": "Asia-Pacific", "USA": "North America", "Canada": "North America",
          "Israel": "Middle East", "Saudi Arabia": "Middle East", "Lebanon": "Middle East",
          "United Arab Emirates": "Middle East", "Bahrain": "Middle East",
          "Brazil": "Other", "RSA": "Other", "Unspecified": "Other"}


# ----------------------------------------------------------------------------- 1. COLLECTION
def load_raw(src) -> pd.DataFrame:
    """src = path, file-like or raw bytes of the Kaggle 'data.csv'."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    df = pd.read_csv(src, encoding="ISO-8859-1", dtype={"InvoiceNo": str, "StockCode": str})
    try:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    except ValueError:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    q = {
        "Missing CustomerID": df.CustomerID.isna().sum(),
        "Missing Description": df.Description.isna().sum(),
        "Duplicate rows": df.duplicated().sum(),
        "Cancelled invoices (InvoiceNo starts with 'C')": df.InvoiceNo.str.startswith("C").sum(),
        "Quantity <= 0": (df.Quantity <= 0).sum(),
        "UnitPrice <= 0": (df.UnitPrice <= 0).sum(),
        "Non-product StockCodes (POST, M, BANK CHARGES...)": (~df.StockCode.str.match(r"^\d{5}")).sum(),
    }
    out = pd.DataFrame({"Issue": list(q), "Rows": [int(v) for v in q.values()]})
    out["% of rows"] = (out.Rows / len(df) * 100).round(2)
    return out


MISSING_ACTION = {"CustomerID": "Rows dropped (customer-level analysis needs an ID)",
                  "Description": "Filled with the standard description of the same StockCode"}


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({"Column": df.columns, "Missing": df.isna().sum().values})
    t["% missing"] = (t.Missing / len(df) * 100).round(2)
    t["Action"] = t.Column.map(MISSING_ACTION).fillna("-")
    return t


def replacement_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Counts of values that were changed (not removed) in rows that survive filtering."""
    df, _, _ = _filter_invalid(raw)
    rows = []

    def add(issue, n, action):
        rows.append({"Issue": issue, "Values affected": int(n), "Action / replacement": action})

    stripped = df.Description.str.strip().str.upper()
    sc = df.StockCode.str.upper()
    std = stripped.dropna().groupby(sc).agg(lambda s: s.mode().iat[0])
    add("Missing Description", df.Description.isna().sum(), "Filled with the most frequent description of that StockCode")
    add("Description with extra spaces / lower-case", (df.Description.notna() & (df.Description != stripped)).sum(),
        "Trimmed and converted to UPPER CASE")
    add("Description differs from the standard one for its StockCode",
        (stripped.notna() & (stripped != sc.map(std))).sum(), "Replaced with the standard description")
    main_country = df.groupby("CustomerID").Country.agg(lambda s: s.mode().iat[0])
    add("Customer appears under more than one Country", (df.Country != df.CustomerID.map(main_country)).sum(),
        "Replaced with the customer's most frequent Country")
    for c in ["Quantity", "UnitPrice"]:
        hi = df[c].quantile(OUTLIER_Q)
        hi = int(hi) if c == "Quantity" else hi
        add(f"{c} outliers (above {OUTLIER_Q:.1%} percentile)", (df[c] > hi).sum(), f"Capped at {hi:g}")
    add("CustomerID stored as decimal (e.g. 17850.0)", len(df), "Converted to integer")
    add("InvoiceDate stored as text", len(raw), "Converted to datetime")
    add("TotalPrice column missing", len(df), "Created as Quantity x UnitPrice")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2-3. CLEANING
def _filter_invalid(raw: pd.DataFrame):
    """Row-level filtering shared by clean() and replacement_report()."""
    log = []

    def step(name, d):
        log.append({"Step": name, "Rows remaining": len(d)})

    step("Raw data", raw)
    df = raw.drop_duplicates()
    step("Remove duplicate rows", df)
    is_c = df.InvoiceNo.str.startswith("C")
    cancels = df[is_c & df.CustomerID.notna()].copy()
    cancels["CustomerID"] = cancels.CustomerID.astype(int)
    df = df[~is_c]
    step("Remove cancelled invoices", df)
    df = df.dropna(subset=["CustomerID"])
    step("Drop rows without CustomerID", df)
    df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]
    step("Remove Quantity/UnitPrice <= 0", df)
    df = df[df.StockCode.str.match(r"^\d{5}")].copy()
    step("Remove non-product StockCodes", df)
    return df, cancels, pd.DataFrame(log)


def clean(raw: pd.DataFrame):
    """Returns (clean_df, cancellations_df, step_log, outlier_caps)."""
    df, cancels, log = _filter_invalid(raw)
    df["CustomerID"] = df.CustomerID.astype(int)
    df["StockCode"] = df.StockCode.str.upper()
    # Missing / inconsistent descriptions -> one standard description per StockCode (most frequent)
    df["Description"] = df.Description.str.strip().str.upper()
    mode = (df.dropna(subset=["Description"]).groupby("StockCode").Description
            .agg(lambda s: s.mode().iat[0]))
    df["Description"] = df.StockCode.map(mode).fillna("UNKNOWN")

    caps = []
    for c in ["Quantity", "UnitPrice"]:
        hi = df[c].quantile(OUTLIER_Q)
        if c == "Quantity":
            hi = int(hi)
        caps.append({"Column": c, "Cap value": hi, "Rows capped": int((df[c] > hi).sum())})
        df[c] = df[c].clip(upper=hi)
    df["Quantity"] = df.Quantity.astype(int)
    return df.reset_index(drop=True), cancels, log, pd.DataFrame(caps)


# ----------------------------------------------------------------------------- 4-5. INTEGRATION / TRANSFORM / REDUCTION
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalPrice"] = df.Quantity * df.UnitPrice
    main_country = df.groupby("CustomerID").Country.agg(lambda s: s.mode().iat[0])
    df["Country"] = df.CustomerID.map(main_country)           # one country per customer
    df["Region"] = df.Country.map(lambda c: "UK" if c == "United Kingdom" else NON_EU.get(c, "Europe"))
    return df


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PriceBand"] = pd.cut(df.UnitPrice, [0, 1, 3, 10, np.inf],
                             labels=["Budget (<=1)", "Economy (1-3)", "Standard (3-10)", "Premium (>10)"])
    df["QtyBand"] = pd.cut(df.Quantity, [0, 1, 6, 12, np.inf],
                           labels=["Single", "Small (2-6)", "Medium (7-12)", "Bulk (13+)"])
    return df


def reduce(df: pd.DataFrame):
    """Data reduction: drop helper columns, downcast numerics, categorical encoding."""
    before = df.memory_usage(deep=True).sum() / 1e6
    out = df.copy()
    for c in ["Country", "Region", "PriceBand", "QtyBand"]:
        out[c] = out[c].astype("category")
    out["CustomerID"] = out.CustomerID.astype("int32")
    out["Quantity"] = pd.to_numeric(out.Quantity, downcast="integer")
    after = out.memory_usage(deep=True).sum() / 1e6
    return out, {"before_mb": round(before, 1), "after_mb": round(after, 1)}


# ----------------------------------------------------------------------------- 6. CONSISTENCY
def consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def chk(name, ok, detail=""):
        rows.append({"Check": name, "Status": "PASS" if ok else "FAIL", "Detail": detail})

    key = ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "CustomerID", "Country"]
    chk("No missing values in key columns", df[key].isna().sum().sum() == 0)
    chk("Quantity and UnitPrice strictly positive", bool(((df.Quantity > 0) & (df.UnitPrice > 0)).all()))
    chk("TotalPrice = Quantity x UnitPrice", np.allclose(df.TotalPrice, df.Quantity * df.UnitPrice))
    n = int((df.groupby("StockCode").Description.nunique() > 1).sum())
    chk("Each StockCode has exactly one Description", n == 0, f"{n} violations")
    n = int((df.groupby("CustomerID").Country.nunique() > 1).sum())
    chk("Each customer belongs to exactly one Country", n == 0, f"{n} violations")
    n = int((df.groupby("InvoiceNo").CustomerID.nunique() > 1).sum())
    chk("Each invoice belongs to exactly one customer", n == 0, f"{n} violations")
    chk("No duplicate rows", int(df.duplicated().sum()) == 0)
    chk("Dates form a valid range", df.InvoiceDate.notna().all(),
        f"{df.InvoiceDate.min():%Y-%m-%d} to {df.InvoiceDate.max():%Y-%m-%d}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 7. STAR SCHEMA
def build_star_schema(df: pd.DataFrame) -> dict:
    df = df.astype({"Country": str, "Region": str})
    d = pd.DataFrame({"Date": np.sort(df.InvoiceDate.dt.normalize().unique())})
    dim_date = d.assign(
        DateKey=d.Date.dt.strftime("%Y%m%d").astype(int), Year=d.Date.dt.year,
        Quarter="Q" + d.Date.dt.quarter.astype(str), Month=d.Date.dt.month,
        YearMonth=d.Date.dt.strftime("%Y-%m"), MonthName=d.Date.dt.month_name(),
        DayName=d.Date.dt.day_name(), IsWeekend=d.Date.dt.dayofweek >= 5)

    dim_geography = df[["Country", "Region"]].drop_duplicates().sort_values("Country").reset_index(drop=True)
    dim_geography.insert(0, "GeoKey", range(1, len(dim_geography) + 1))

    cust = df.groupby("CustomerID").agg(TotalSpend=("TotalPrice", "sum"),
                                        FirstPurchaseDate=("InvoiceDate", "min")).reset_index()
    cust["SpendBand"] = pd.qcut(cust.TotalSpend, 4, labels=["Low", "Medium", "High", "Top"]).astype(str)
    dim_customer = cust.rename(columns={"CustomerID": "CustomerKey"}).drop(columns="TotalSpend")

    prod = df.groupby("StockCode").agg(Description=("Description", "first"),
                                       AvgPrice=("UnitPrice", "median")).reset_index()
    prod["PriceBand"] = pd.cut(prod.AvgPrice, [0, 1, 3, 10, np.inf],
                               labels=["Budget (<=1)", "Economy (1-3)", "Standard (3-10)", "Premium (>10)"]).astype(str)
    prod.insert(0, "ProductKey", range(1, len(prod) + 1))
    dim_product = prod

    f = (df.assign(DateKey=df.InvoiceDate.dt.strftime("%Y%m%d").astype(int))
           .merge(dim_product[["StockCode", "ProductKey"]], on="StockCode")
           .merge(dim_geography[["Country", "GeoKey"]], on="Country")
           .rename(columns={"CustomerID": "CustomerKey"}))
    fact_sales = f[["InvoiceNo", "InvoiceDate", "DateKey", "CustomerKey", "ProductKey", "GeoKey",
                    "Quantity", "UnitPrice", "TotalPrice"]].reset_index(drop=True)
    return {"fact_sales": fact_sales, "dim_date": dim_date, "dim_customer": dim_customer,
            "dim_product": dim_product, "dim_geography": dim_geography}


def save_to_sqlite(tables: dict, path: str = "warehouse.db") -> None:
    with sqlite3.connect(path) as con:
        for name, t in tables.items():
            t.to_sql(name, con, if_exists="replace", index=False)


# ----------------------------------------------------------------------------- 8-9. OLAP & CUBE
def olap_base(t: dict) -> pd.DataFrame:
    """Join the fact table with all dimensions (the 'cube' working table)."""
    out = (t["fact_sales"]
           .merge(t["dim_date"].drop(columns="Date"), on="DateKey")
           .merge(t["dim_customer"], on="CustomerKey")
           .merge(t["dim_product"].drop(columns="AvgPrice"), on="ProductKey")
           .merge(t["dim_geography"], on="GeoKey"))
    for c in out.select_dtypes("object"):
        out[c] = out[c].astype("category")
    return out


HIERARCHIES = {"Time": ["Year", "Quarter", "YearMonth"],
               "Geography": ["Region", "Country"],
               "Product": ["PriceBand", "Description"]}


def roll_up(base, hierarchy, depth, measure="TotalPrice"):
    """Aggregate to the first `depth` levels of a hierarchy (fewer levels = rolled up)."""
    cols = hierarchy[:depth]
    return base.groupby(cols, observed=True)[measure].sum().reset_index()


def drill_down(base, hierarchy, depth, measure="TotalPrice"):
    """One level more detailed than `depth`."""
    return roll_up(base, hierarchy, min(depth + 1, len(hierarchy)), measure)


def slice_cube(base, dim, value):
    return base[base[dim].astype(str) == str(value)]


def dice_cube(base, conditions: dict):
    mask = pd.Series(True, index=base.index)
    for dim, values in conditions.items():
        mask &= base[dim].astype(str).isin([str(v) for v in values])
    return base[mask]


def pivot(base, index, columns, measure="TotalPrice", agg="sum"):
    return base.pivot_table(index=index, columns=columns, values=measure, aggfunc=agg,
                            observed=True, fill_value=0)


def data_cube(base, dims, measure="TotalPrice"):
    """Full data cube: every one of the 2^n group-bys, 'ALL' marks a rolled-up dimension."""
    b = base[list(dims) + [measure]].copy()
    b[list(dims)] = b[list(dims)].astype(str)
    parts = []
    for r in range(len(dims) + 1):
        for combo in combinations(dims, r):
            if combo:
                g = b.groupby(list(combo))[measure].agg(["sum", "count"]).reset_index()
            else:
                g = pd.DataFrame({"sum": [b[measure].sum()], "count": [len(b)]})
            for d in dims:
                if d not in combo:
                    g[d] = "ALL"
            parts.append(g[list(dims) + ["sum", "count"]])
    return pd.concat(parts, ignore_index=True).rename(columns={"sum": "Revenue", "count": "Lines"})


# ----------------------------------------------------------------------------- 10-11. CHURN MODEL
def build_features(df, cancels, cutoff):
    """RFM-style behaviour features using ONLY data up to `cutoff` (no leakage)."""
    hist = df[df.InvoiceDate <= cutoff]
    f = hist.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"), FirstPurchase=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"), Monetary=("TotalPrice", "sum"),
        TotalItems=("Quantity", "sum"), UniqueProducts=("StockCode", "nunique"))
    f["Recency"] = (cutoff - f.LastPurchase).dt.days
    f["TenureDays"] = (cutoff - f.FirstPurchase).dt.days
    f["AvgBasketValue"] = f.Monetary / f.Frequency
    f["AvgDaysBetweenOrders"] = f.TenureDays / f.Frequency
    recent = hist[hist.InvoiceDate > cutoff - pd.Timedelta(days=90)].groupby("CustomerID").InvoiceNo.nunique()
    f["OrdersLast90d"] = recent.reindex(f.index).fillna(0)
    ret = cancels[cancels.InvoiceDate <= cutoff].groupby("CustomerID").InvoiceNo.nunique()
    f["ReturnCount"] = ret.reindex(f.index).fillna(0)
    return f.drop(columns=["LastPurchase", "FirstPurchase"])


def build_churn_dataset(df, cancels, horizon=CHURN_HORIZON_DAYS):
    """Features from before the cutoff; label Churn=1 if the customer never buys after it."""
    cutoff = df.InvoiceDate.max() - pd.Timedelta(days=horizon)
    f = build_features(df, cancels, cutoff)
    active_later = set(df.loc[df.InvoiceDate > cutoff, "CustomerID"])
    f["Churn"] = (~f.index.isin(active_later)).astype(int)
    return f, cutoff


def train_and_evaluate(data: pd.DataFrame, seed: int = 42) -> dict:
    X, y = data.drop(columns="Churn"), data["Churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
    }
    rows, fitted, roc, cms = [], {}, {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred, proba = m.predict(Xte), m.predict_proba(Xte)[:, 1]
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred),
                     "Precision": precision_score(yte, pred), "Recall": recall_score(yte, pred),
                     "F1": f1_score(yte, pred), "ROC-AUC": roc_auc_score(yte, proba)})
        fitted[name] = m
        fpr, tpr, _ = roc_curve(yte, proba)
        roc[name], cms[name] = (fpr, tpr), confusion_matrix(yte, pred)
    res = pd.DataFrame(rows).round(4)
    best = res.sort_values("ROC-AUC", ascending=False).Model.iloc[0]
    imp = pd.Series(fitted["Random Forest"].feature_importances_, index=X.columns).sort_values(ascending=False)
    return {"results": res, "models": fitted, "best": best, "roc": roc, "cms": cms,
            "importance": imp, "columns": list(X.columns), "n_train": len(Xtr), "n_test": len(Xte)}


def score_customers(df, cancels, model, columns):
    """Churn probability for every customer as of the last date in the data."""
    f = build_features(df, cancels, df.InvoiceDate.max())
    f["ChurnProbability"] = model.predict_proba(f[columns])[:, 1]
    f["Risk"] = pd.cut(f.ChurnProbability, [-0.01, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])
    return f.sort_values("ChurnProbability", ascending=False)


# ----------------------------------------------------------------------------- 12. INSIGHTS
def business_insights(base, data, ml, scores) -> list[str]:
    rev = base.TotalPrice.sum()
    by_c = base.groupby("Country", observed=True).TotalPrice.sum().sort_values(ascending=False)
    uk = by_c.get("United Kingdom", 0) / rev
    top_export = ", ".join(by_c.drop("United Kingdom", errors="ignore").head(3).index)
    m = base.groupby("YearMonth", observed=True).TotalPrice.sum()
    d = base.groupby("DayName", observed=True).TotalPrice.sum().sort_values(ascending=False)
    weekend = base.loc[base.IsWeekend, "TotalPrice"].sum() / rev
    cust_rev = base.groupby("CustomerKey").TotalPrice.sum().sort_values(ascending=False)
    top10 = cust_rev.head(max(1, len(cust_rev) // 10)).sum() / rev
    best = ml["results"].set_index("Model").loc[ml["best"]]
    high = scores[scores.Risk == "High"]
    return [
        f"The UK generates {uk:.0%} of revenue; the biggest export markets are {top_export}.",
        f"Revenue peaks in {m.idxmax()} (£{m.max():,.0f}); plan stock and campaigns ahead of the seasonal peak.",
        f"{d.index[0]} is the strongest trading day; weekends contribute only {weekend:.0%} of revenue.",
        f"The top 10% of customers account for {top10:.0%} of revenue, so retention of this group matters most.",
        f"{data.Churn.mean():.0%} of customers who were active before the cut-off made no purchase in the following {CHURN_HORIZON_DAYS} days (churned).",
        f"Best model: {ml['best']} (ROC-AUC {best['ROC-AUC']:.2f}, recall {best['Recall']:.2f}, precision {best['Precision']:.2f}).",
        f"Strongest churn drivers: {', '.join(ml['importance'].head(3).index)}.",
        f"{len(high):,} customers are currently High risk and hold £{high.Monetary.sum():,.0f} of historical revenue: target them with win-back offers first.",
    ]


# ----------------------------------------------------------------------------- RUN EVERYTHING
def run_pipeline(src, clean_out=None) -> dict:
    """src = path to the Kaggle data.csv; clean_out = optional path to save the cleaned data."""
    raw = load_raw(src)
    n_raw, quality_raw = len(raw), quality_report(raw)
    raw_sample = raw.head(10)
    raw_info = pd.DataFrame({"dtype": raw.dtypes.astype(str), "missing": raw.isna().sum(),
                             "missing %": (raw.isna().mean() * 100).round(2), "unique": raw.nunique()})
    raw_describe = raw[["Quantity", "UnitPrice"]].describe().round(2)
    missing_raw, replaced = missing_table(raw), replacement_report(raw)
    df, cancels, log, caps = clean(raw)
    del raw
    df = discretize(transform(df))
    df, memory = reduce(df)
    checks = consistency_checks(df)
    if clean_out:
        df.to_csv(clean_out, index=False)
    stats = {"rows": len(df), "customers": df.CustomerID.nunique(), "products": df.StockCode.nunique(),
             "revenue": float(df.TotalPrice.sum()), "start": df.InvoiceDate.min(), "end": df.InvoiceDate.max(),
             "price_bands": df.PriceBand.value_counts(sort=False), "qty_bands": df.QtyBand.value_counts(sort=False)}
    tables = build_star_schema(df)
    base = olap_base(tables)
    data, cutoff = build_churn_dataset(df, cancels)
    ml = train_and_evaluate(data)
    scores = score_customers(df, cancels, ml["models"][ml["best"]], ml["columns"])
    return {"raw_sample": raw_sample, "raw_info": raw_info, "raw_describe": raw_describe,
            "clean_sample": df.head(10), "clean_saved_to": clean_out,
            "n_raw": n_raw, "quality_raw": quality_raw, "missing_raw": missing_raw,
            "replaced": replaced, "log": log, "caps": caps, "memory": memory,
            "checks": checks, "stats": stats, "tables": tables, "base": base, "data": data, "cutoff": cutoff,
            "ml": ml, "scores": scores, "insights": business_insights(base, data, ml, scores)}
