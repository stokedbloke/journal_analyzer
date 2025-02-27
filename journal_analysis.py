import pandas as pd
from collections import Counter

# Load the CSV file
csv_path = "analyzed_journal_entries.csv"
df = pd.read_csv(csv_path)

# Convert date fields to datetime objects
df['date_created'] = pd.to_datetime(df['date_created'])
df['date_edited'] = pd.to_datetime(df['date_edited'])

# Extract day-of-week from creation date
df['day_of_week'] = df['date_created'].dt.day_name()


# Compute the time difference (in days) between modified and created dates
df['diff_days'] = (df['date_edited'] - df['date_created']).dt.total_seconds() / (3600 * 24)

# Mark whether an entry was modified on a different day
df['modified'] = df['diff_days'] > 0


# Analysis 1: Basic modification stats
modified_entries = df[df['diff_days'] > 0]
num_modified = len(modified_entries)

print(f"Total entries: {len(df)}")
print(f"Entries modified on a different day: {num_modified}")

# Descriptive stats for modification time spread
spread_stats = modified_entries['diff_days'].describe()
print("\nTime spread (in days) between creation and modification for modified entries:")
print(spread_stats)


# --- Insight 1: Frequency of Dominant Emotions ---
emotion_counts = df['dominant_emotion'].value_counts()
print("Dominant Emotion Counts:")
print(emotion_counts)

# Compute percentage of entries labeled 'neutral'
neutral_percentage = (emotion_counts.get("neutral", 0) / len(df)) * 100
print(f"\nPercentage of neutral entries: {neutral_percentage:.2f}%")

# --- Insight 2: Modification Statistics ---
# Determine if an entry was modified on a different day (compare date portions)
df['modified'] = df.apply(lambda row: row['date_created'].date() != row['date_edited'].date(), axis=1)
modified_counts = df['modified'].value_counts()
print("\nModified vs. Unmodified Entries:")
print(modified_counts)

# --- Insight 3: Average Dominant Score ---
overall_avg_score = df['dominant_score'].mean()
print(f"\nOverall average dominant score: {overall_avg_score:.3f}")

# Average dominant score by dominant emotion
avg_score_by_emotion = df.groupby('dominant_emotion')['dominant_score'].mean()
print("\nAverage Dominant Score by Emotion:")
print(avg_score_by_emotion)

# Average dominant score by day of week
avg_score_by_day = df.groupby('day_of_week')['dominant_score'].mean().sort_index()
print("\nAverage Dominant Score by Day of Week:")
print(avg_score_by_day)

# --- Insight 4: Top Keywords ---
# Assuming the keywords column is a comma-separated string of keywords
all_keywords = df['keywords'].dropna().str.cat(sep=", ")
keyword_list = [kw.strip() for kw in all_keywords.split(",") if kw.strip()]
keyword_counts = Counter(keyword_list)
print("\nTop 10 Keywords:")
for kw, count in keyword_counts.most_common(10):
    print(f"{kw}: {count}")

# --- Insight 5: Modified Entries and Non-Neutral Sentiment ---
modified_nonneutral = df[(df['modified'] == True) & (df['dominant_emotion'] != 'neutral')]
num_modified = modified_counts.get(True, 0)
perc_modified_nonneutral = (len(modified_nonneutral) / num_modified * 100) if num_modified > 0 else 0
print(f"\nPercentage of modified entries that are non-neutral: {perc_modified_nonneutral:.2f}%")
