
from dotenv import load_dotenv
import os

load_dotenv("journal_analyzer.env")  # Explicitly specify path

NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_API_TOKEN = os.getenv("NOTION_API_TOKEN")

if NOTION_API_TOKEN is None or NOTION_DATABASE_ID is None:
    raise ValueError("Missing Notion API token or database ID. Check your .env file.")


from notion_client import Client
notion = Client(auth=NOTION_API_TOKEN)

#for cleaining
import re
import json
import pandas as pd


def clean_text(text):
    """Removes excessive whitespace and normalizes text."""
    text = text.strip()  # Remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.lower()  # Convert to lowercase for consistency


def clean_entries(entries):
    """Cleans journal entries and removes empty ones."""
    cleaned_entries = []
    for entry in entries:
        title = clean_text(entry["title"])
        content = clean_text(entry["content"])

        if content:  # Ensure we keep only entries with meaningful content
            cleaned_entries.append({
                "title": title,
                "content": content,
                "date_created": entry["date_created"],
                "date_edited": entry["date_edited"],
                "date_property": entry["date_property"]
            })
    
    return cleaned_entries


def save_to_csv(entries, filename="journal_entries.csv"):
    df = pd.DataFrame(entries)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Data saved to {filename}")

def save_to_json(entries, filename="journal_entries.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")

import sqlite3

def save_to_sqlite(entries, db_filename="journal_data.db"):
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # Drop existing table if it exists
    cursor.execute("DROP TABLE IF EXISTS journal")

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            date_created TEXT,
            date_edited TEXT,
            date_property TEXT,
            sentiment TEXT,
            sentiment_score FLOAT,
            keywords TEXT
        )
    """)  # Closing the SQL string properly


    for entry in entries:
        # Insert data
        keywords_str = ", ".join(entry["keywords"]) if isinstance(entry["keywords"], list) else entry["keywords"]

        cursor.execute("""
            INSERT INTO journal (title, content, date_created, date_edited, date_property, sentiment, sentiment_score, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (entry["title"], entry["content"], entry["date_created"], entry["date_edited"], entry["date_property"], entry["sentiment"], entry["sentiment_score"], keywords_str))

    conn.commit()
    conn.close()
    print(f"Data saved to {db_filename}")

# List all databases the integration can access
#databases = notion.search(filter={"property": "object", "value": "database"})
#print(f"Notion Databases: {databases}\n")


def fetch_database_properties(database_id):
    response = notion.databases.retrieve(database_id=database_id)
    print("Database Properties:")
    for key, value in response["properties"].items():
        print(f"{key}: {value['type']}")

def fetch_page_content(block_id):
    response = notion.blocks.children.list(block_id=block_id)
    content = []

    for block in response['results']:
        # Check if the block contains rich text
        if block['type'] in ['paragraph', 'bulleted_list_item', 'heading_1', 'heading_2', 'heading_3']:
            rich_text = block[block['type']].get('rich_text', [])
            if rich_text:
                # Combine plain text from all rich_text segments
                text = "".join([text['plain_text'] for text in rich_text])
                if block['type'] == 'bulleted_list_item':
                    content.append(f"- {text}")  # Format as a list item
                elif block['type'] == 'heading_1':
                    content.append(f"# {text}")  # Format as a heading
                elif block['type'] == 'heading_2':
                    content.append(f"## {text}")  # Format as a heading
                elif block['type'] == 'heading_3':
                    content.append(f"### {text}")  # Format as a heading
                else:
                    content.append(text)  # Default formatting for paragraphs
        else:
            # Keep track of skipped block types
            skipped_types = set()

            for block in response['results']:
                if block['type'] not in ['paragraph', 'bulleted_list_item', 'heading_1', 'heading_2', 'heading_3']:
                    if block['type'] not in skipped_types:
                        '''print(f"Skipping block type: {block['type']}")'''
                        skipped_types.add(block['type'])


    return "\n".join(content) if content else "No content available"




def fetch_journal_entries(database_id):
    response = notion.databases.query(database_id=database_id)
    results = response["results"]

    journal_entries = []
    for entry in results:
        properties = entry["properties"]
        block_id = entry["id"]  # Use the entry ID as block_id for fetching content

        # Fetch the page content
        content = fetch_page_content(block_id) # or "No content available"

        # Append the entry
        journal_entries.append({
            "title": properties["Name"]["title"][0]["plain_text"] if properties["Name"]["title"] else "Untitled",
            "content": content,
            "date_created": entry["created_time"],  # Metadata field for date created
            "date_edited": entry["last_edited_time"],  # Metadata field for date last edited
            "date_property": properties["Date"]["date"]["start"] if properties["Date"]["date"] else None
        })

    return journal_entries


# We'll use a pretrained model from transformers for sentiment analysis.
# This function will return "POSITIVE", "NEGATIVE", or "NEUTRAL" with a confidence score.
from transformers import pipeline

# Load the sentiment analysis model
# PyTorch (default), change to "tf" if you're using TensorFlow
# Explicitly specify the model and revision
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f", framework="pt")
# Ensure PyTorch is used

import textwrap

# Instead of cutting off content, split long text into smaller chunks and analyze each separately.
def analyze_sentiment(text, chunk_size=512):
    """Splits long text into chunks and returns an average sentiment score."""
    if not text.strip():
        return {"label": "neutral", "score": 0.0}

    chunks = textwrap.wrap(text, chunk_size)
    scores = []
    labels = []

    for chunk in chunks:
        result = sentiment_model(chunk)
        scores.append(result[0]["score"])
        labels.append(result[0]["label"])

    # Majority vote for sentiment label
    final_label = max(set(labels), key=labels.count)
    avg_score = sum(scores) / len(scores)  # Average confidence score

    return {"label": final_label, "score": avg_score}



# We'll use TF-IDF (Term Frequency-Inverse Document Frequency) to extract important words. 
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
from nltk.corpus import stopwords

# Fix: Convert stop_words to a list instead of a set
stop_words = list(stopwords.words("english"))  # ✅ Convert set to list

def extract_keywords(text, top_n=5):
    """Extracts top keywords from text using TF-IDF."""
    if not text.strip():  # Handle empty text cases
        return []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)  # ✅ Use 'english' instead of custom stopwords
    tfidf_matrix = vectorizer.fit_transform([text])

    feature_names = vectorizer.get_feature_names_out()
    return list(feature_names)



import time
from tqdm import tqdm

# Modify your journal processing function to include sentiment and keyword analysis:
def analyze_journal_entries(entries):
    """Enhances journal entries with sentiment analysis and keyword extraction."""
    analyzed_entries = []
    
    total_entries = len(entries)    
    start_time = time.time()  # Start the timer

    with tqdm(total=total_entries, desc="Processing Entries", unit="entry") as pbar:
        for i, entry in enumerate(entries):
            sentiment = analyze_sentiment(entry["content"])
            keywords = extract_keywords(entry["content"])

            analyzed_entries.append({
                "title": entry["title"],
                "content": entry["content"],
                "date_created": entry["date_created"],
                "date_edited": entry["date_edited"],
                "date_property": entry["date_property"],
                "sentiment": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "keywords": keywords
            })

            pbar.update(1)  # Update progress bar

            # Estimate remaining time
            elapsed_time = time.time() - start_time
            avg_time_per_entry = elapsed_time / (i + 1)
            estimated_total_time = avg_time_per_entry * total_entries
            remaining_time = estimated_total_time - elapsed_time

            pbar.set_postfix({
                "Elapsed": f"{elapsed_time:.1f}s",
                "ETA": f"{remaining_time:.1f}s"
            })
    
    print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")
    return analyzed_entries


def save_to_csv(entries, filename="analyzed_journal_entries.csv"):
    df = pd.DataFrame(entries)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Data saved to {filename}")

def save_to_json(entries, filename="analyzed_journal_entries.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")


# Fetch all journal entries
entries = fetch_journal_entries(NOTION_DATABASE_ID)

# Clean the data
cleaned_entries = clean_entries(entries)

# Analyze sentiment & keywords
analyzed_entries = analyze_journal_entries(cleaned_entries)

# Save the data to different formats
save_to_csv(analyzed_entries)
save_to_json(analyzed_entries)
save_to_sqlite(analyzed_entries)

print (f"Total entries: {len(entries)}")

'''for entry in entries:
    print(f"Title: {entry['title']}")
    print(f"Content: {entry['content'][:100]}...")  # Show a snippet of content
    print(f"Created: {entry['date_created']}")
    print(f"Last Edited: {entry['date_edited']}")
    print(f"Date Property: {entry['date_property']}")
    print("---")'''
