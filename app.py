from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DATA_PATH as CFG_PATH
from src import pipeline as P

st.set_page_config(page_title="Retail Analytics & Churn", page_icon="🛒", layout="wide")
BASE_DIR = Path(__file__).parent
# ---------------- dataset path comes from config.py ----------------
DATA_PATH = Path(CFG_PATH) if Path(CFG_PATH).is_absolute() else BASE_DIR / CFG_PATH   # set in config.py
CLEAN_PATH = BASE_DIR / "data" / "cleaned_data.csv"  # cleaned dataset is saved here
# ---------------------------------------------------------------------------------------------

if not DATA_PATH.exists():
    st.error(f"Dataset not found at: {DATA_PATH}\n\nDownload data.csv from Kaggle and set DATA_PATH in config.py.")
    st.stop()


@st.cache_resource(show_spinner="Cleaning data, building warehouse and training models (about a minute on first load)...")
def load_all(path: str, clean_out: str):
    return P.run_pipeline(path, clean_out)


st.sidebar.title("🛒 Retail Analytics")
try:
    R = load_all(str(DATA_PATH), str(CLEAN_PATH))
except OSError:   # read-only disk on some hosts: skip saving the cleaned CSV
    R = load_all(str(DATA_PATH), "")
base, S = R["base"], R["stats"]

page = st.sidebar.radio("Section", ["1 · Data & Quality", "2 · Data Warehouse", "3 · OLAP Operations",
                                    "4 · Data Cube", "5 · Churn Prediction", "6 · Business Insights"])

# ------------------------------------------------------------------ 1
if page.startswith("1"):
    st.title("Data Understanding, Cleaning & Quality")
    c = st.columns(5)
    c[0].metric("Raw rows", f"{R['n_raw']:,}")
    c[1].metric("Clean rows", f"{S['rows']:,}")
    c[2].metric("Customers", f"{S['customers']:,}")
    c[3].metric("Products", f"{S['products']:,}")
    c[4].metric("Revenue", f"£{S['revenue']:,.0f}")
    st.caption(f"Period: {S['start']:%d %b %Y} to {S['end']:%d %b %Y}")

    st.subheader("1 · Data collection & understanding")
    st.caption(f"Source file: `{DATA_PATH.name}`")
    st.dataframe(R["raw_sample"], width="stretch", hide_index=True)
    a, b = st.columns([3, 2])
    a.dataframe(R["raw_info"], width="stretch")
    b.dataframe(R["raw_describe"], width="stretch")

    st.subheader("Missing values in the raw data")
    st.dataframe(R["missing_raw"], width="stretch", hide_index=True)

    st.subheader("Issues found in the raw data")
    st.dataframe(R["quality_raw"], width="stretch", hide_index=True)

    st.subheader("Cleaning pipeline")
    fig = px.bar(R["log"], x="Rows remaining", y="Step", orientation="h")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Values replaced / fixed (rows kept, values changed)")
    st.dataframe(R["replaced"], width="stretch", hide_index=True)

    a, b = st.columns(2)
    a.subheader("Outlier handling")
    a.caption(f"Values above the {P.OUTLIER_Q:.1%} percentile are capped (winsorising).")
    a.dataframe(R["caps"], hide_index=True, width="stretch")
    b.subheader("Data reduction")
    b.write(f"Memory: **{R['memory']['before_mb']} MB → {R['memory']['after_mb']} MB** "
            "(helper columns dropped, numeric downcasting, categorical encoding).")

    st.subheader("Discretization")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(S["price_bands"], title="Unit price bands", labels={"value": "Rows", "index": ""}),
                   width="stretch")
    b.plotly_chart(px.bar(S["qty_bands"], title="Order quantity bands", labels={"value": "Rows", "index": ""}),
                   width="stretch")

    st.subheader("Consistency checks")
    st.dataframe(R["checks"], hide_index=True, width="stretch")

    st.subheader("Cleaned & transformed data (preview)")
    st.dataframe(R["clean_sample"], hide_index=True, width="stretch")
    if R["clean_saved_to"]:
        st.success(f"Full cleaned dataset saved to: {R['clean_saved_to']}")

# ------------------------------------------------------------------ 2
elif page.startswith("2"):
    st.title("Data Warehouse – Star Schema")
    dot = r'''digraph G { rankdir=LR; node [shape=record, style=filled, fillcolor="#e8f0fe"];
    fact [fillcolor="#fde7c8", label="{fact_sales|InvoiceNo\lDateKey (FK)\lCustomerKey (FK)\lProductKey (FK)\lGeoKey (FK)\lQuantity\lUnitPrice\lTotalPrice\l}"];
    dim_date [label="{dim_date|DateKey (PK)\lDate\lYear\lQuarter\lMonth\lYearMonth\lDayName\lIsWeekend\l}"];
    dim_customer [label="{dim_customer|CustomerKey (PK)\lFirstPurchaseDate\lSpendBand\l}"];
    dim_product [label="{dim_product|ProductKey (PK)\lStockCode\lDescription\lAvgPrice\lPriceBand\l}"];
    dim_geography [label="{dim_geography|GeoKey (PK)\lCountry\lRegion\l}"];
    dim_date -> fact; dim_customer -> fact; dim_product -> fact; dim_geography -> fact; }'''
    st.graphviz_chart(dot)
    st.markdown("**Grain:** one row per invoice line (one product on one invoice). "
                "**Measures:** Quantity, UnitPrice, TotalPrice.")
    T = R["tables"]
    st.dataframe(pd.DataFrame({"Table": list(T), "Rows": [len(t) for t in T.values()],
                               "Columns": [", ".join(t.columns) for t in T.values()]}),
                 hide_index=True, width="stretch")
    name = st.selectbox("Preview a table", list(T))
    st.dataframe(T[name].head(200), width="stretch")

# ------------------------------------------------------------------ 3
elif page.startswith("3"):
    st.title("OLAP Operations")
    tabs = st.tabs(["Roll-up / Drill-down", "Slice", "Dice", "Pivot"])

    with tabs[0]:
        h_name = st.radio("Hierarchy", list(P.HIERARCHIES), horizontal=True)
        hier = P.HIERARCHIES[h_name]
        level = st.select_slider("Level of detail (left = rolled up, right = drilled down)",
                                 options=list(range(1, len(hier) + 1)), format_func=lambda i: hier[i - 1])
        out = P.roll_up(base, hier, level)
        st.caption(f"Grouped by: {' → '.join(hier[:level])}")
        out["Label"] = out[hier[:level]].astype(str).agg(" · ".join, axis=1)
        shown = out.nlargest(25, "TotalPrice") if len(out) > 40 else out
        st.plotly_chart(px.bar(shown, x="Label", y="TotalPrice"), width="stretch")
        st.dataframe(out.drop(columns="Label"), width="stretch", hide_index=True)

    with tabs[1]:
        dim = st.selectbox("Slice on dimension", ["Year", "Quarter", "Region", "Country", "DayName", "PriceBand", "SpendBand"])
        val = st.selectbox("Value", sorted(map(str, base[dim].unique())))
        sl = P.slice_cube(base, dim, val)
        c = st.columns(3)
        c[0].metric("Revenue", f"£{sl.TotalPrice.sum():,.0f}")
        c[1].metric("Customers", f"{sl.CustomerKey.nunique():,}")
        c[2].metric("Invoices", f"{sl.InvoiceNo.nunique():,}")
        m = sl.groupby("YearMonth", observed=True).TotalPrice.sum().reset_index()
        st.plotly_chart(px.line(m, x="YearMonth", y="TotalPrice", markers=True, title=f"{dim} = {val}"),
                        width="stretch")

    with tabs[2]:
        conds = {}
        a, b, c = st.columns(3)
        conds["Year"] = a.multiselect("Year", sorted(map(str, base.Year.unique())), default=sorted(map(str, base.Year.unique())))
        conds["Region"] = b.multiselect("Region", sorted(map(str, base.Region.unique())), default=sorted(map(str, base.Region.unique())))
        conds["SpendBand"] = c.multiselect("SpendBand", sorted(map(str, base.SpendBand.unique())), default=sorted(map(str, base.SpendBand.unique())))
        dc = P.dice_cube(base, conds)
        st.metric("Revenue in selected sub-cube", f"£{dc.TotalPrice.sum():,.0f}")
        g = dc.groupby(["Region", "SpendBand"], observed=True).TotalPrice.sum().reset_index()
        st.plotly_chart(px.bar(g, x="Region", y="TotalPrice", color="SpendBand", barmode="group"),
                        width="stretch")

    with tabs[3]:
        dims = ["Year", "Quarter", "YearMonth", "DayName", "Region", "Country", "PriceBand", "SpendBand"]
        a, b, c, d = st.columns(4)
        idx = a.selectbox("Rows", dims, index=4)
        col = b.selectbox("Columns", dims, index=7)
        meas = c.selectbox("Measure", ["TotalPrice", "Quantity"])
        agg = d.selectbox("Aggregate", ["sum", "mean", "count"])
        pt = P.pivot(base, idx, col, meas, agg)
        st.dataframe(pt.round(0), width="stretch")
        st.plotly_chart(px.imshow(pt, aspect="auto", color_continuous_scale="Blues"), width="stretch")

# ------------------------------------------------------------------ 4
elif page.startswith("4"):
    st.title("Data Cube")
    all_dims = ["Year", "Quarter", "Region", "Country", "DayName", "PriceBand", "SpendBand"]
    dims = st.multiselect("Cube dimensions (a cube with n dimensions has 2ⁿ cuboids)", all_dims,
                          default=["Year", "Region", "SpendBand"], max_selections=5)
    if dims:
        @st.cache_data(show_spinner="Building cube...")
        def build_cube(d: tuple):
            return P.data_cube(base, list(d))

        cube = build_cube(tuple(dims))
        st.caption(f"{len(cube):,} cells in {2 ** len(dims)} cuboids. 'ALL' = dimension rolled up.")
        cols = st.columns(len(dims))
        view = cube
        for cc, d in zip(cols, dims):
            choice = cc.selectbox(d, ["(any)"] + sorted(cube[d].unique()))
            if choice != "(any)":
                view = view[view[d] == choice]
        st.dataframe(view.sort_values("Revenue", ascending=False), width="stretch", hide_index=True)
        st.download_button("Download cube as CSV", cube.to_csv(index=False), "data_cube.csv")

# ------------------------------------------------------------------ 5
elif page.startswith("5"):
    ml, data = R["ml"], R["data"]
    st.title("Customer Churn Prediction")
    st.info(f"**Churn definition:** a customer active before {R['cutoff']:%d %b %Y} who makes **no purchase in the "
            f"following {P.CHURN_HORIZON_DAYS} days**. Features use only data before that date (no leakage). "
            f"Churn rate: **{data.Churn.mean():.1%}** of {len(data):,} customers. "
            f"Train/test split: {ml['n_train']:,} / {ml['n_test']:,} (stratified).")
    st.dataframe(ml["results"], hide_index=True, width="stretch")
    st.success(f"Best model by ROC-AUC: **{ml['best']}**")

    t1, t2, t3, t4 = st.tabs(["ROC curve", "Confusion matrix", "Feature importance", "At-risk customers"])
    with t1:
        fig = go.Figure()
        for n, (fpr, tpr) in ml["roc"].items():
            fig.add_scatter(x=fpr, y=tpr, name=n)
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash"))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, width="stretch")
    with t2:
        mname = st.radio("Model", list(ml["cms"]), index=list(ml["cms"]).index(ml["best"]), horizontal=True)
        cm = pd.DataFrame(ml["cms"][mname], index=["Actual: Active", "Actual: Churn"],
                          columns=["Pred: Active", "Pred: Churn"])
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues"), width="stretch")
    with t3:
        st.plotly_chart(px.bar(ml["importance"].sort_values(), orientation="h",
                               labels={"value": "Importance", "index": ""}), width="stretch")
    with t4:
        sc = R["scores"]
        a, b = st.columns(2)
        risk = a.multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
        top = b.slider("Max customers", 10, 500, 100)
        view = sc[sc.Risk.isin(risk)].head(top).reset_index().round(2)
        st.dataframe(view, hide_index=True, width="stretch")
        st.download_button("Download list (CSV)", view.to_csv(index=False), "at_risk_customers.csv")
        cid = st.number_input("Look up a CustomerID", value=int(sc.index[0]), step=1)
        if cid in sc.index:
            r = sc.loc[cid]
            st.metric(f"Customer {cid} churn probability", f"{r.ChurnProbability:.0%}", help=f"Risk: {r.Risk}")
        else:
            st.warning("Customer not found.")

# ------------------------------------------------------------------ 6
else:
    st.title("Business Insights & Conclusion")
    for line in R["insights"]:
        st.markdown(f"- {line}")
    st.subheader("Recommendations")
    st.markdown("""
- **Win-back campaign:** contact High-risk customers, ranked by historical revenue, with personalised offers.
- **Early-warning rule:** flag customers whose days since last order exceed their usual gap between orders.
- **Protect the top spenders:** loyalty perks and account management for the Top spend band.
- **Seasonal planning:** build stock and run campaigns ahead of the peak months shown in the roll-up.
- **Grow exports:** replicate what works in the strongest non-UK markets.
""")
    st.subheader("Limitations")
    st.markdown("- Only about 25% of rows have no CustomerID, so guest purchases are excluded.\n"
                "- Data covers one year only, so yearly seasonality is seen once.\n"
                "- The data has no demographics, so the model relies on purchase behaviour only.")
