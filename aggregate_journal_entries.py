import pandas as pd
import sqlite3
import json
import numpy as np

def custom_daily_aggregator(df_group):
    """
    Receives a sub-DataFrame for a single date_created group.
    Returns a single-row Series with aggregated values across all columns.
    """
    result = {}
    # All rows in this group have the same date_created, so take the first one
    date_val = df_group["date_created"].iloc[0]
    result["date_created"] = date_val

    for col in df_group.columns:
        if col == "date_created":
            continue

        # If the column is numeric (e.g., emotion columns), take the mean.
        if pd.api.types.is_numeric_dtype(df_group[col]):
            result[col] = df_group[col].mean(skipna=True)
        else:
            if col == "journal_id":
                joined_ids = ", ".join(df_group["journal_id"].dropna().unique())
                result["journal_id"] = joined_ids
            elif col == "keywords":
                all_keywords = []
                for val in df_group["keywords"].dropna():
                    if isinstance(val, list):
                        all_keywords.extend(val)
                    else:
                        all_keywords.append(str(val))
                all_keywords = list(set(all_keywords))
                result["keywords"] = " | ".join(all_keywords)
            else:
                unique_vals = df_group[col].dropna().astype(str).unique()
                result[col] = " | ".join(unique_vals)
    return pd.Series(result)

# ------------------- MAIN EXECUTION -------------------

# Load your DataFrame from "analyzed_journal_entries.csv"
df_csv = pd.read_csv("analyzed_journal_entries.csv")

# Convert date_created to datetime and truncate to YYYY-MM-DD
df_csv["date_created"] = pd.to_datetime(df_csv["date_created"]).dt.date

# Group by date_created and apply the custom aggregator.
# Passing group_keys=False to avoid including grouping columns.
aggregated_df = df_csv.groupby("date_created", as_index=False, group_keys=False).apply(custom_daily_aggregator)

# ------------------- SAVE TO CSV -------------------
csv_path = "aggregated_journal_entries.csv"
aggregated_df.to_csv(csv_path, index=False)
print(f"✅ Aggregated journal data saved to CSV: {csv_path}")

# ------------------- SAVE TO JSON -------------------
# Convert DataFrame to a list of dictionaries.
records = aggregated_df.to_dict(orient="records")
json_path = "aggregated_journal_entries.json"
with open(json_path, "w", encoding="utf-8") as f:
    # default=str converts non-serializable objects (like date) to string.
    json.dump(records, f, ensure_ascii=False, indent=2, default=str)
print(f"✅ Aggregated journal data saved to JSON: {json_path}")

# ------------------- SAVE TO SQLITE -------------------
db_path = "journal_data.db"
conn = sqlite3.connect(db_path)
aggregated_df.to_sql("AggregatedJournalEntries", conn, if_exists="replace", index=False)
conn.close()
print(f"✅ Aggregated journal data saved to SQLite: {db_path}")
