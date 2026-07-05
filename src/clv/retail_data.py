from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clv.config import DEMO_DATA_FILE, RAW_FILE_CANDIDATES, RAW_DIR, RANDOM_STATE


COLUMN_ALIASES = {
    "invoice": "InvoiceNo",
    "invoiceno": "InvoiceNo",
    "stockcode": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoicedate": "InvoiceDate",
    "price": "UnitPrice",
    "unitprice": "UnitPrice",
    "customerid": "CustomerID",
    "customer id": "CustomerID",
    "country": "Country",
}

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

PRODUCTS = [
    ("10001", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
    ("10002", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("10003", "JUMBO BAG RED RETROSPOT", 1.95),
    ("10004", "PARTY BUNTING", 4.95),
    ("10005", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.25),
    ("10006", "LUNCH BAG CARS BLUE", 1.65),
    ("10007", "ALARM CLOCK BAKELIKE RED", 3.75),
    ("10008", "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("10009", "WOODEN PICTURE FRAME WHITE FINISH", 2.10),
    ("10010", "VINTAGE SNAP CARDS", 0.85),
    ("10011", "HAND WARMER UNION JACK", 2.10),
    ("10012", "PAPER CHAIN KIT VINTAGE CHRISTMAS", 2.95),
]

COUNTRIES = [
    "United Kingdom",
    "Germany",
    "France",
    "EIRE",
    "Netherlands",
    "Spain",
    "Portugal",
    "Belgium",
]

CATEGORY_RULES = [
    ("Bags & Storage", ["BAG", "BASKET", "BOX", "HOLDER", "STORAGE", "WALLET", "PURSE"]),
    ("Kitchen & Tableware", ["CAKE", "CUP", "PLATE", "BOWL", "MUG", "TEA", "COFFEE", "CUTLERY", "NAPKIN"]),
    ("Home Decor", ["HEART", "CANDLE", "LIGHT", "LANTERN", "FRAME", "CLOCK", "SIGN", "CUSHION", "WALL"]),
    ("Stationery & Cards", ["CARD", "PAPER", "PENCIL", "PEN", "NOTEBOOK", "STICKER", "TAPE", "WRAP"]),
    ("Gifts & Novelty", ["GIFT", "TOY", "DOLL", "CHARM", "KEY", "MAGNET", "RETROSPOT", "VINTAGE"]),
    ("Seasonal", ["CHRISTMAS", "EASTER", "HALLOWEEN", "VALENTINE", "PARTY", "BUNTING"]),
    ("Kids & Baby", ["BABY", "CHILD", "GIRL", "BOY", "LUNCH", "SCHOOL"]),
    ("Accessories", ["SCARF", "NECKLACE", "BRACELET", "RING", "HAIR", "MIRROR"]),
]


def find_raw_data_file(raw_dir: Path = RAW_DIR) -> Path | None:
    for path in RAW_FILE_CANDIDATES:
        if path.exists():
            return path

    if raw_dir.exists():
        candidates = sorted(
            list(raw_dir.glob("*.xlsx"))
            + list(raw_dir.glob("*.xls"))
            + list(raw_dir.glob("*.csv"))
        )
        non_demo = [path for path in candidates if "demo" not in path.name.lower()]
        if non_demo:
            return non_demo[0]
        if candidates:
            return candidates[0]

    return None


def create_demo_data(path: Path = DEMO_DATA_FILE, n_customers: int = 900) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    start = pd.Timestamp("2010-01-01")
    end = pd.Timestamp("2011-12-09")
    days = (end - start).days

    personas = {
        "champion": (0.14, 1.9, 4.2, 1.25),
        "loyal": (0.26, 1.2, 3.0, 1.05),
        "regular": (0.28, 0.65, 2.2, 1.00),
        "new": (0.10, 0.9, 1.7, 0.95),
        "bargain": (0.12, 0.5, 1.7, 0.82),
        "dormant": (0.10, 0.35, 1.5, 0.90),
    }
    names = list(personas)
    probs = np.array([personas[name][0] for name in names])
    probs = probs / probs.sum()

    rows = []
    invoice = 530000
    for idx in range(n_customers):
        customer_id = 12000 + idx
        persona = rng.choice(names, p=probs)
        _, monthly_orders, avg_qty, price_mult = personas[persona]
        country = rng.choice(COUNTRIES, p=[0.74, 0.07, 0.06, 0.04, 0.03, 0.025, 0.02, 0.015])

        if persona == "new":
            first_day = int(rng.integers(int(days * 0.72), days - 20))
            last_day = days
        elif persona == "dormant":
            first_day = int(rng.integers(0, int(days * 0.35)))
            last_day = int(rng.integers(first_day + 20, int(days * 0.65)))
        else:
            first_day = int(rng.integers(0, int(days * 0.65)))
            last_day = days if rng.random() > 0.25 else int(rng.integers(first_day + 30, days))

        active_months = max(1.0, (last_day - first_day) / 30)
        order_count = max(1, int(rng.poisson(monthly_orders * active_months)))
        order_days = np.sort(rng.integers(first_day, last_day + 1, size=order_count))

        for order_day in order_days:
            invoice += 1
            invoice_date = start + pd.Timedelta(days=int(order_day), hours=int(rng.integers(8, 21)))
            line_count = int(rng.integers(1, 5))
            is_cancel = rng.random() < 0.035

            for _ in range(line_count):
                stock_code, description, base_price = PRODUCTS[int(rng.integers(0, len(PRODUCTS)))]
                quantity = max(1, int(rng.poisson(avg_qty)))
                if is_cancel:
                    quantity = -quantity
                price = max(0.2, base_price * price_mult * rng.normal(1.0, 0.08))
                rows.append(
                    {
                        "Invoice": f"{'C' if is_cancel else ''}{invoice}",
                        "StockCode": stock_code,
                        "Description": description,
                        "Quantity": quantity,
                        "InvoiceDate": invoice_date,
                        "Price": round(float(price), 2),
                        "Customer ID": customer_id,
                        "Country": country,
                    }
                )

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_transactions(data_file: Path | str | None = None, allow_demo_data: bool = True) -> pd.DataFrame:
    if data_file is None:
        data_file = find_raw_data_file()

    if data_file is None:
        if not allow_demo_data:
            raise FileNotFoundError("No UCI Online Retail II file found in data/raw.")
        data_file = create_demo_data()

    path = Path(data_file)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        raw = pd.concat(sheets.values(), ignore_index=True)
    else:
        raw = pd.read_csv(path, encoding_errors="ignore")

    return clean_transactions(raw)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    rename_map = {}
    for column in data.columns:
        normalized = str(column).strip().lower().replace("_", " ").replace("-", " ")
        normalized_compact = normalized.replace(" ", "")
        if normalized in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[normalized]
        elif normalized_compact in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[normalized_compact]

    data = data.rename(columns=rename_map)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required Online Retail II columns: {missing}")

    data = data[REQUIRED_COLUMNS].copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data["CustomerID"] = data["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True)
    data["InvoiceNo"] = data["InvoiceNo"].astype(str).str.strip()
    data["StockCode"] = data["StockCode"].astype(str).str.strip()
    data["Description"] = data["Description"].fillna("Unknown Product").astype(str).str.strip()
    data["Country"] = data["Country"].fillna("Unknown").astype(str).str.strip()

    data = data.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
    data = data[(data["CustomerID"].str.lower() != "nan") & (data["UnitPrice"] > 0)]
    data["is_cancelled"] = data["InvoiceNo"].str.upper().str.startswith("C") | (data["Quantity"] < 0)
    data["line_revenue"] = data["Quantity"] * data["UnitPrice"]
    data["gross_revenue"] = data["line_revenue"].clip(lower=0)
    data["return_revenue"] = (-data["line_revenue"]).clip(lower=0)
    data["net_revenue"] = data["line_revenue"]
    data["product_category"] = data["Description"].map(assign_product_category)

    return data.sort_values("InvoiceDate").reset_index(drop=True)


def assign_product_category(description: str) -> str:
    text = str(description).upper()
    for category, keywords in CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return "General Merchandise"
