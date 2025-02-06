from dotenv import load_dotenv
import os

print(f"Loading environment from: journal_analyzer.env")
load_dotenv("journal_analyzer.env")

# Debug: Print masked versions of credentials
api_token = os.getenv("NOTION_API_TOKEN")
db_id = os.getenv("NOTION_DATABASE_ID")
print(f"API Token loaded: {'secret_' + '*'*10 if api_token and api_token.startswith('secret_') else 'INVALID'}")
print(f"Database ID loaded: {db_id[:4] + '*'*8 if db_id else 'MISSING'}")

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
    """Mild cleaning: Normalize spaces but keep capitalization & formatting."""
    if not text:
        return text  # Ensure we don't mistakenly process NoneType

    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.lower()  # Convert to lowercase for consistency
    return text.strip()  # Remove leading/trailing whitespace but keep everything else

"""Cleans journal entries and removes empty ones."""
def clean_entries(entries):
    cleaned_entries = []
    for entry in entries:
        title = clean_text(entry["title"])
        content = clean_text(entry["content"])

        if content:  # Ensure we keep only entries with meaningful content
            cleaned_entries.append({
                "title": title,
                "content": content,
                "date_created": entry["date_created"],
                "date_edited": entry["date_edited"]
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

    # Create table with additional emotion columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            date_created TEXT,
            date_edited TEXT,
            primary_emotion TEXT,
            primary_emotion_score FLOAT,
            secondary_emotion TEXT,
            secondary_emotion_score FLOAT,
            tertiary_emotion TEXT,
            tertiary_emotion_score FLOAT,
            keywords TEXT
        )
    """)

    for entry in entries:
        # Assuming entry["sentiment"] now contains the emotion analysis results
        emotions = entry["sentiment"]["top_emotions"]
        cursor.execute("""
            INSERT INTO journal (
                title, content, date_created, date_edited,
                primary_emotion, primary_emotion_score,
                secondary_emotion, secondary_emotion_score,
                tertiary_emotion, tertiary_emotion_score,
                keywords
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry["title"], entry["content"],
            entry["date_created"], entry["date_edited"],
            emotions[0]["emotion"], emotions[0]["score"],
            emotions[1]["emotion"], emotions[1]["score"],
            emotions[2]["emotion"], emotions[2]["score"],
            ", ".join(entry["keywords"])
        ))

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


def fetch_page_content(page_id):
    """Fetches full content of a Notion page by recursively retrieving all blocks including nested ones."""
    try:
        # Track skipped block types at the outer scope
        skipped_types = set()

        # Get page title
        page = notion.pages.retrieve(page_id)
        page_title = page["properties"]["Name"]["title"][0]["plain_text"] if page["properties"]["Name"]["title"] else "Untitled"

        def get_block_content(block_id):
            blocks = notion.blocks.children.list(block_id=block_id)
            content = []
            
            for block in blocks.get("results", []):
                block_type = block["type"]
                
                # Handle text-based block types
                if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", 
                                "bulleted_list_item", "numbered_list_item", "quote", 
                                "callout", "toggle", "to_do"]:
                    
                    # Get the rich text content from the block
                    rich_text = block[block_type].get("rich_text", [])
                    if rich_text:
                        text = "".join([t["plain_text"] for t in rich_text])
                        
                        # Format based on block type
                        if block_type.startswith("heading_"):
                            level = block_type[-1]  # Get heading level (1, 2, or 3)
                            content.append(f"{'#' * int(level)} {text}")
                        elif block_type == "bulleted_list_item":
                            content.append(f"- {text}")
                        elif block_type == "numbered_list_item":
                            content.append(f"1. {text}")
                        elif block_type == "quote":
                            content.append(f"> {text}")
                        elif block_type == "callout":
                            content.append(f"[!] {text}")
                        elif block_type == "to_do":
                            checked = block["to_do"].get("checked", False)
                            content.append(f"[{'x' if checked else ' '}] {text}")
                        else:  # paragraph and others
                            content.append(text)
                        
                        # Handle nested blocks
                        if block.get("has_children", False):
                            nested_content = get_block_content(block["id"])
                            # Indent nested content
                            content.extend("  " + line for line in nested_content.split("\n") if line)
                
                # Handle table blocks
                elif block_type == "table":
                    # Get table rows
                    table_rows = []
                    table_block_id = block["id"]
                    table_children = notion.blocks.children.list(block_id=table_block_id)
                    
                    for row_block in table_children.get("results", []):
                        if row_block["type"] == "table_row":
                            # Extract cells from the row
                            row_cells = []
                            for cell in row_block["table_row"]["cells"]:
                                cell_text = "".join([t["plain_text"] for t in cell]) if cell else ""
                                row_cells.append(cell_text)
                            table_rows.append(row_cells)
                    
                    if table_rows:
                        # Calculate column widths
                        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table_rows)]
                        
                        # Format table with ASCII borders
                        # Header row
                        header_row = table_rows[0]
                        content.append("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
                        content.append("|" + "|".join(f" {cell:<{width}} " for cell, width in zip(header_row, col_widths)) + "|")
                        content.append("+" + "+".join("=" * (width + 2) for width in col_widths) + "+")
                        
                        # Data rows
                        for row in table_rows[1:]:
                            content.append("|" + "|".join(f" {str(cell):<{width}} " for cell, width in zip(row, col_widths)) + "|")
                            content.append("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
                        
                        content.append("")  # Add blank line after table
                
                # Track unsupported block types
                else:
                    if block_type not in skipped_types:
                        skipped_types.add(block_type)

            return "\n".join(content)

        full_content = get_block_content(page_id)
        if skipped_types:
            print(f"📝 Skipped block types in '{page_title}': {sorted(skipped_types)}")
        return full_content

    except Exception as e:
        print(f"🚨 ERROR fetching page content for '{page_title}': {e}")
        return ""


def fetch_journal_entries(database_id):
    """Fetch all journal entries from the Notion database, handling pagination."""
    entries = []
    has_more = True
    next_cursor = None

    while has_more:
        params = {"page_size": 100}  # Limit per request
        if next_cursor:
            params["start_cursor"] = next_cursor  # Use next_cursor if available

        response = notion.databases.query(database_id=database_id, **params)

        # Process results
        for result in response.get("results", []):
            properties = result["properties"]
            block_id = result["id"]  # Use the entry ID as block_id for fetching content
            # Fetch the page content
            content = fetch_page_content(result["id"]) or ""
            entries.append({
                "title": properties["Name"]["title"][0]["plain_text"] if properties["Name"]["title"] else "Untitled",
                "content": content,
                "date_created": properties.get("Created", {}).get("created_time", ""),
                "date_edited": properties.get("Last Edited", {}).get("last_edited_time", ""),
            })

        # Check if there are more pages
        has_more = response.get("has_more", False)
        next_cursor = response.get("next_cursor", None)

    return entries


# We'll use a pretrained model from transformers for sentiment analysis.
# This function will return "POSITIVE", "NEGATIVE", or "NEUTRAL" with a confidence score.
from transformers import pipeline

# Load the sentiment analysis model
# PyTorch (default), change to "tf" if you're using TensorFlow
# Explicitly specify the model and revision
sentiment_model = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None  # This replaces return_all_scores=True
)
# Ensure PyTorch is used

import textwrap

# Instead of cutting off content, split long text into smaller chunks and analyze each separately.
def analyze_sentiment(text, chunk_size=512):
    """Splits long text into chunks and returns an average sentiment score."""
    if not text.strip():
        return {"label": "neutral", "score": 0.0, "top_emotions": [
            {"emotion": "neutral", "score": 0.0},
            {"emotion": "neutral", "score": 0.0},
            {"emotion": "neutral", "score": 0.0}
        ]}

    chunks = textwrap.wrap(text, chunk_size)
    all_emotions = []  # Initialize the list here

    for chunk in chunks:
        results = sentiment_model(chunk)[0]  # Get all emotion scores
        # Sort emotions by score
        sorted_emotions = sorted(results, key=lambda x: x['score'], reverse=True)
        # Take top 3 emotions
        top_emotions = sorted_emotions[:3]
        all_emotions.append(top_emotions)

    # Aggregate emotions across chunks
    final_emotions = {}
    for chunk_emotions in all_emotions:
        for emotion in chunk_emotions:
            label = emotion['label']
            score = emotion['score']
            if label in final_emotions:
                final_emotions[label] += score
            else:
                final_emotions[label] = score

    # Average the scores and find dominant emotion
    for label in final_emotions:
        final_emotions[label] /= len(chunks)
    
    dominant_emotion = max(final_emotions.items(), key=lambda x: x[1])

    # Format the return value to match what save_to_sqlite expects
    return {
        "label": dominant_emotion[0],
        "score": dominant_emotion[1],
        "top_emotions": [
            {"emotion": e, "score": s} 
            for e, s in sorted(final_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
    }



# We'll use TF-IDF (Term Frequency-Inverse Document Frequency) to extract important words. 
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

import re

nltk.download("stopwords")
from nltk.corpus import stopwords

# Fix: Convert stop_words to a list instead of a set
stop_words = list(stopwords.words("english"))  # ✅ Convert set to list

def extract_keywords(text, top_n=5):
    """Extracts top keywords from text using TF-IDF."""
    #text = clean_text(text)  # Apply cleaning
    if not text.strip():  # Handle empty text cases
        print("⚠️ Warning: Text became empty after cleaning.", repr(text))
        return ["NO_KEYWORDS_FOUND"]
    #print(f"🔍 Processing text: {repr(text)[:100]}...")  # Print first 100 chars for debugging

    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)  # ✅ Use 'english' instead of custom stopwords
    try:
        
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        keywords = feature_names[:10] if len(feature_names) > 0 else ["NO_KEYWORDS_FOUND"]# Return top 10 words
        return list(keywords)
    except ValueError as e:
        print(f"🚨 ERROR: Could not process text: {repr(text)}")
        return []


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
            if not entry["content"].strip():
                print("⚠️ Skipping empty entry:", entry["title"])
                continue
            keywords = extract_keywords(entry["content"])

            analyzed_entries.append({
                "title": entry["title"],
                "content": entry["content"],
                "date_created": entry["date_created"],
                "date_edited": entry["date_edited"],
                "sentiment": sentiment,  # Now passing the full sentiment object
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
print(f"Total entries fetched: {len(entries)}")

# Clean the data
cleaned_entries = clean_entries(entries)

# Analyze sentiment & keywords
analyzed_entries = analyze_journal_entries(cleaned_entries)

# Save the data to different formats
save_to_csv(analyzed_entries)
save_to_json(analyzed_entries)
save_to_sqlite(analyzed_entries)

print (f"Total entries: {len(entries)}")