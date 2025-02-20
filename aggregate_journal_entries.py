import pandas as pd
import sqlite3

# Load CSV
csv_path = "analyzed_journal_entries.csv"
df = pd.read_csv(csv_path)

# Ensure date is in proper format (truncate to YYYY-MM-DD)
df['date_created'] = pd.to_datetime(df['date_created']).dt.date

# Aggregate by date
aggregated_df = df.groupby('date_created').agg({
    'title': lambda x: ' | '.join(x.dropna()),
    'content': lambda x: ' '.join(x.dropna()),
    'keywords': lambda x: ', '.join(x.dropna()),
    'dominant_emotion': lambda x: x.mode()[0] if not x.isna().all() else None,
    'dominant_score': 'mean',
    'emotion_1': lambda x: x.mode()[0] if not x.isna().all() else None,
    'emotion_1_score': 'mean',
    'emotion_2': lambda x: x.mode()[0] if not x.isna().all() else None,
    'emotion_2_score': 'mean',
    'emotion_3': lambda x: x.mode()[0] if not x.isna().all() else None,
    'emotion_3_score': 'mean'
}).reset_index()

# Save to a new CSV file
aggregated_csv_path = "aggregated_journal_entries.csv"
aggregated_df.to_csv(aggregated_csv_path, index=False)
print(f"✅ Aggregated journal data saved to: {aggregated_csv_path}")

# Save to SQLite
db_path = "journal_data.db"
conn = sqlite3.connect(db_path)
aggregated_df.to_sql("AggregatedJournalEntries", conn, if_exists="replace", index=False)
conn.close()
print(f"✅ Aggregated data saved to SQLite: {db_path}")
