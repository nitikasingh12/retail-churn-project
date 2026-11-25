"""Preprocessing report: what was wrong with the raw data and exactly how it was fixed.

Run:  python preprocessing.py
Creates in ./outputs :  cleaned_data.csv, cleaning_report.txt, cleaning_summary.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_PATH as CFG_PATH
from src import pipeline as P

BASE_DIR = Path(__file__).parent
DATA_PATH = Path(CFG_PATH) if Path(CFG_PATH).is_absolute() else BASE_DIR / CFG_PATH
OUT = BASE_DIR / "outputs"
OUT.mkdir(exist_ok=True)
lines: list[str] = []


def show(text=""):
    print(text)
    lines.append(str(text))


def section(title):
    show("\n" + "=" * 80)
    show(title)
    show("=" * 80)


def table(df):
    show(df.to_string(index=False))


def main():
    if not Path(DATA_PATH).exists():
        raise SystemExit(f"Dataset not found at {DATA_PATH!r}. Edit DATA_PATH in config.py")

    # ---------------------------------------------------------------- 1. understanding
    raw = P.load_raw(DATA_PATH)
    n_raw = len(raw)
    section("1. DATA COLLECTION & UNDERSTANDING")
    show(f"File        : {DATA_PATH}")
    show(f"Rows x cols : {raw.shape[0]:,} x {raw.shape[1]}")
    show(f"Period      : {raw.InvoiceDate.min():%d %b %Y} to {raw.InvoiceDate.max():%d %b %Y}")
    show(f"Customers   : {raw.CustomerID.nunique():,} | Products: {raw.StockCode.nunique():,} | "
         f"Countries: {raw.Country.nunique()} | Invoices: {raw.InvoiceNo.nunique():,}")
    table(pd.DataFrame({"Column": raw.columns, "Type": raw.dtypes.astype(str).values,
                        "Example": [str(raw[c].dropna().iloc[0])[:35] for c in raw.columns]}))
    show("")
    show(raw[["Quantity", "UnitPrice"]].describe().round(2).to_string())

    # ---------------------------------------------------------------- 2-3. problems found
    section("2. MISSING VALUES IN THE RAW DATA")
    table(P.missing_table(raw))

    section("3. DATA QUALITY PROBLEMS IN THE RAW DATA")
    quality = P.quality_report(raw)
    table(quality)

    # ---------------------------------------------------------------- run the cleaning
    df, cancels, log, caps = P.clean(raw)
    replaced = P.replacement_report(raw)
    missing_before = int(raw.isna().sum().sum())
    dup_before = int(raw.duplicated().sum())
    bad_before = int(((raw.Quantity <= 0) | (raw.UnitPrice <= 0)).sum())
    df = P.discretize(P.transform(df))

    section("4. CLEANING STEPS - ROWS REMOVED")
    log = log.copy()
    log["Rows removed"] = log["Rows remaining"].diff().abs().fillna(0).astype(int)
    log["% of raw data"] = (log["Rows removed"] / n_raw * 100).round(2)
    table(log)
    show(f"\nKept {len(df):,} of {n_raw:,} rows ({len(df) / n_raw:.1%}); removed {n_raw - len(df):,}.")

    section("5. VALUES REPLACED / FIXED (rows kept, values changed)")
    table(replaced)
    show("\nOutliers (values above the cap were replaced by the cap):")
    table(caps)

    section("6. BEFORE vs AFTER")
    ba = pd.DataFrame({
        "Metric": ["Rows", "Missing values (all cells)", "Duplicate rows", "Cancelled invoices",
                   "Rows with Quantity<=0 or UnitPrice<=0", "Max Quantity", "Max UnitPrice"],
        "Before": [n_raw, missing_before, dup_before, int(raw.InvoiceNo.str.startswith("C").sum()), bad_before,
                   raw.Quantity.max(), raw.UnitPrice.max()],
        "After": [len(df), int(df[["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice",
                                  "CustomerID", "Country"]].isna().sum().sum()),
                  int(df.duplicated().sum()), int(df.InvoiceNo.str.startswith("C").sum()),
                  int(((df.Quantity <= 0) | (df.UnitPrice <= 0)).sum()), df.Quantity.max(), df.UnitPrice.max()],
    })
    fmt = lambda x: f"{x:,.2f}" if isinstance(x, float) and not float(x).is_integer() else f"{int(x):,}"
    ba[["Before", "After"]] = ba[["Before", "After"]].astype(object).map(fmt)
    table(ba)

    section("7. CONSISTENCY CHECKS ON THE CLEANED DATA")
    table(P.consistency_checks(df))

    # ---------------------------------------------------------------- save
    df.to_csv(OUT / "cleaned_data.csv", index=False)
    (OUT / "cleaning_report.txt").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].barh(log.Step, log["Rows remaining"])
    ax[0].invert_yaxis()
    ax[0].set(title="Rows remaining after each cleaning step", xlabel="Rows")
    ax[1].barh(quality.Issue, quality.Rows, color="tab:red")
    ax[1].invert_yaxis()
    ax[1].set(title="Problems found in the raw data", xlabel="Rows affected")
    plt.tight_layout()
    fig.savefig(OUT / "cleaning_summary.png", dpi=150)
    show(f"\nSaved to {OUT}: cleaned_data.csv, cleaning_report.txt, cleaning_summary.png")


if __name__ == "__main__":
    main()
