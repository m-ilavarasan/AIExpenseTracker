import json
import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
from pydantic import BaseModel
import streamlit as st

# Load environment variables from .env file if present
load_dotenv(override=True)

# --- CONFIGURATION ---
CACHE_FILE = "merchant_cache.json"
CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Utilities & Bills",
    "Investments & SIP",
    "Travel & Fuel",
    "Shopping",
    "Rent & Housing",
    "Medical & Health",
    "Personal Transfer",
    "Other",
    "Other Invest",
    "Fees",
]

# Fast local keyword rules (handles ~70% of Indian narrations in < 0.1s with 0 API calls)
KEYWORD_RULES = {
    "Investments & SIP": [
        "INDIAN CLEARING CORP",
        "ZERODHA",
        "GROWW",
        "KUVERA",
        "CAMS",
        "KFINTECH",
        "ANGELONE",
        "UPSTOX",
        "HSBC",
        "LIC",
        "TATA AIG",
    ],
    "Utilities & Bills": [
        "AIRTEL",
        "JIO",
        "ACT FIBERNET",
        "TNEB",
        "BESCOM",
        "BROADBAND",
        "ELECTRICITY",
        "BBPS",
        "WATER",
        "GAS",
        "INDANE",
        "INDIAN OIL",
    ],
    "Food & Dining": [
        "SWIGGY",
        "ZOMATO",
        "STARBUCKS",
        "MCDONALD",
        "DOMINOS",
        "KFC",
        "CAFE",
        "RESTAURANT",
        "BAKERY",
        "BIRYANI",
        "PALM",
    ],
    "Groceries": [
        "BLINKIT",
        "ZEPTO",
        "INSTAMART",
        "BIGBASKET",
        "DMART",
        "SUPERMARKET",
        "GROCERY",
        "VEGETABLE",
        "MILK",
        "JEYACHANDRAN RETAIL",
        "VINAYAGA",
    ],
    "Travel & Fuel": [
        "UBER",
        "OLA",
        "RAPIDO",
        "IRCTC",
        "REDBUS",
        "INDIGO",
        "PETROL",
        "HPCL",
        "BPCL",
        "IOCL",
        "SHELL",
        "FASTAG",
        "VETRIVEL",
        "VIGNESH",
        "INDIAN RAILWAY",
    ],
    "Shopping": [
        "AMAZON",
        "FLIPKART",
        "MYNTRA",
        "AJIO",
        "DECATHLON",
        "MEESHO",
        "NYKAA",
    ],
    "Medical & Health": [
        "PHARMACY",
        "MEDICAL",
        "HOSPITAL",
        "CLINIC",
        "APOLLO",
        "MEDPLUS",
        "1MG",
        "PHARMEASY",
        "SRM",
        "PADMA",
        "VIJAYA",
    ],
    "Rent & Housing": ["PRABHAKAR"],
    "Other Invest": ["DHANYA", "SAATVIK", "NAVAJEE"],
    "Fees": ["AKG"],
}


# Pydantic schema for Structured Outputs
class MerchantCategory(BaseModel):
  merchant_key: str
  category: str


class BatchClassification(BaseModel):
  results: list[MerchantCategory]


# --- 1. TOKENIZER & PRIVACY SANITIZER ---
def extract_clean_merchant(narration: str) -> str:
  """Extracts merchant/payee info and strips sensitive UTR/Reference numbers."""
  if not isinstance(narration, str):
    return "UNKNOWN"
  text = narration.strip().upper()

  # Pattern 1: UPI-MERCHANT-SUBMERCHANT-VPA-IFSC-REF-REMARKS
  if text.startswith("UPI-"):
    parts = text.split("-")
    payee = parts[1].strip() if len(parts) > 1 else ""
    vpa = parts[3].strip() if len(parts) > 3 and "@" in parts[3] else ""
    remarks = parts[-1].strip() if len(parts) > 5 else ""
    remarks = re.sub(r"\d+", "", remarks).strip()
    return f"{payee} | {vpa} | {remarks}".strip(" |")

  # Pattern 2: ACH D- INDIAN CLEARING CORP-77X59
  if text.startswith("ACH"):
    parts = text.split("-")
    entity = parts[1].strip() if len(parts) > 1 else text
    return f"ACH: {entity}"

  # Pattern 3: NEFT / IMPS / POS
  cleaned = re.sub(r"\d{6,}", "", text)
  return cleaned[:60].strip()


# --- 2. LOCAL CACHE & KEYWORD MATCHER ---
def load_cache() -> dict:
  if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
      return json.load(f)
  return {}


def save_cache(cache: dict):
  with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, indent=2)


def rule_based_match(merchant_token: str) -> str | None:
  for category, keywords in KEYWORD_RULES.items():
    if any(kw in merchant_token for kw in keywords):
      return category
  return None


# --- 3. AI AGENT CATEGORIZER ---
def classify_merchants_with_ai(unknown_keys: list[str], api_key: str) -> dict:
  if not unknown_keys:
    return {}

  client = genai.Client(api_key=api_key)
  prompt = f"""
    You are an expert Indian personal finance assistant.
    Classify each bank narration token into EXACTLY ONE of these categories:
    {', '.join(CATEGORIES)}

    Hints for Indian narrations:
    - 'INDIAN CLEARING CORP', 'ZERODHA', 'GROWW', 'CAMS' -> Investments & SIP
    - 'AIRTEL', 'JIO', 'TNEB', 'BESCOM', 'BROADBAND', 'BBPS' -> Utilities & Bills
    - 'PAYTMQR' or local store names -> infer from name (e.g., Medicals -> Medical & Health, Supermarket/Stores -> Groceries, Cafe/Biryani -> Food & Dining)
    - Individual person names with @okicici/@okaxis/@ybl and no business words -> Personal Transfer

    Tokens to classify:
    {json.dumps(unknown_keys)}
    """

  response = client.models.generate_content(
      model="gemini-3.8-flash",
      contents=prompt,
      config=types.GenerateContentConfig(
          response_mime_type="application/json",
          response_schema=BatchClassification,
          temperature=0.1,
      ),
  )

  parsed = json.loads(response.text)
  return {
      item["merchant_key"]: item["category"] for item in parsed["results"]
  }


# --- 4. STREAMLIT WEB INTERFACE ---
st.set_page_config(page_title="Private AI Expense Tracker", layout="wide")
st.title("Private Bank Statement AI Expense Tracker")

# Sidebar API key setup (reads from environment/.env automatically)
default_key = os.environ.get("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input(
    "Gemini API Key", value=default_key, type="password"
)

uploaded_file = st.file_uploader(
    "Upload Bank Statement (CSV or Excel)", type=["csv", "xlsx", "xls"]
)

if uploaded_file:
  if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
  else:
    df = pd.read_excel(uploaded_file)

  st.subheader("1. Map Your Bank's Columns")
  cols = df.columns.tolist()
  c1, c2, c3 = st.columns(3)
  date_col = c1.selectbox("Date Column", cols, index=0)
  desc_col = c2.selectbox(
      "Description / Particulars Column", cols, index=min(1, len(cols) - 1)
  )
  debit_col = c3.selectbox(
      "Withdrawal / Debit Amount Column", cols, index=min(2, len(cols) - 1)
  )

  if st.button("Analyze & Categorize Expenses"):
    if not api_key:
      st.error(
          "Please enter your Gemini API Key in the left sidebar or set"
          " GEMINI_API_KEY."
      )
      st.stop()

    # Clean numeric debit column
    df["Expense_Amount"] = pd.to_numeric(
        df[debit_col].astype(str).str.replace(",", "").str.strip(),
        errors="coerce",
    ).fillna(0)

    # Filter only actual expenses (Debits > 0)
    expenses_df = df[df["Expense_Amount"] > 0].copy()

    # Tokenize descriptions
    expenses_df["Clean_Merchant"] = expenses_df[desc_col].apply(
        extract_clean_merchant
    )

    # 1. Check local cache first
    cache = load_cache()
    unique_merchants = expenses_df["Clean_Merchant"].unique().tolist()

    # 2. Apply instant local keyword rules
    still_unknown = []
    for m in unique_merchants:
      if m not in cache:
        matched_cat = rule_based_match(m)
        if matched_cat:
          cache[m] = matched_cat
        else:
          still_unknown.append(m)
    save_cache(cache)

    st.info(
        f"Found {len(expenses_df)} debit transactions"
        f" ({len(unique_merchants)} unique merchants). "
        f"{len(unique_merchants) - len(still_unknown)} matched instantly via"
        f" cache/rules; "
        f"{len(still_unknown)} sent to AI Agent."
    )

    # 3. Call AI ONLY for remaining unknowns in batches of 150
    if still_unknown:
      batch_size = 150
      total_batches = (len(still_unknown) + batch_size - 1) // batch_size
      progress_bar = st.progress(
          0, text=f"Classifying {len(still_unknown)} unique merchants..."
      )

      for idx, i in enumerate(range(0, len(still_unknown), batch_size)):
        batch = still_unknown[i : i + batch_size]
        progress_bar.progress(
            idx / total_batches,
            text=f"AI Agent processing batch {idx + 1} of {total_batches}...",
        )
        try:
          new_mappings = classify_merchants_with_ai(batch, api_key)
          cache.update(new_mappings)
          save_cache(cache)
          if idx + 1 < total_batches:
            time.sleep(3)
        except Exception as e:
          st.error(f"Error on batch {idx + 1}: {e}")
          break

      progress_bar.empty()

    # Map categories back to dataframe
    expenses_df["Category"] = (
        expenses_df["Clean_Merchant"].map(cache).fillna("Other")
    )

    # --- DISPLAY RESULTS ---
    st.subheader("2. Expense Summary by Category")
    summary = (
        expenses_df.groupby("Category")["Expense_Amount"]
        .agg(Total_Spent="sum", Transaction_Count="count")
        .reset_index()
        .sort_values(by="Total_Spent", ascending=False)
    )
    summary["% of Spend"] = (
        summary["Total_Spent"] / summary["Total_Spent"].sum() * 100
    ).round(1)

    col_table, col_chart = st.columns(2)
    col_table.dataframe(summary, use_container_width=True, hide_index=True)
    col_chart.bar_chart(summary.set_index("Category")["Total_Spent"])

    st.subheader("3. Detailed Categorized Transactions")
    output_df = expenses_df[
        [date_col, desc_col, "Clean_Merchant", "Category", "Expense_Amount"]
    ]
    st.dataframe(output_df, use_container_width=True, hide_index=True)

    # Export to CSV
    csv_data = output_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Categorized CSV",
        data=csv_data,
        file_name="categorized_expenses.csv",
        mime="text/csv",
    )
