from dotenv import load_dotenv
import os
import re
import json
import pandas as pd
import sqlite3
import time
from tqdm import tqdm
import textwrap
from transformers import pipeline
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

print(f"Loading environment from: journal_analyzer.env")
load_dotenv("journal_analyzer.env", override=True)

NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_API_TOKEN = os.getenv("NOTION_API_TOKEN")
if NOTION_DATABASE_ID:
    NOTION_DATABASE_ID = NOTION_DATABASE_ID.strip("<>")
if NOTION_API_TOKEN:
    NOTION_API_TOKEN = NOTION_API_TOKEN.strip("<>")

if NOTION_API_TOKEN is None or NOTION_DATABASE_ID is None:
    raise ValueError("Missing Notion API token or database ID. Check your .env file.")

from notion_client import Client
notion = Client(auth=NOTION_API_TOKEN)

# ------------------ Cleaning Functions ------------------
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
                "date_edited": entry["date_edited"],
                "journal_id": entry["journal_id"],
            })
    
    return cleaned_entries

# ------------------ Saving Functions ------------------
def save_to_csv(entries, filename="analyzed_journal_entries.csv"):
    """Save entries to CSV with flattened sentiment structure."""
    flattened_entries = []
    
    for entry in entries:
        # Check for missing ID
        if "journal_id" not in entry:
            print("Warning: Missing ID in entry:", entry)  # Debugging line
        
        flat_entry = {
            "title": entry["title"],
            "journal_id": entry.get("journal_id", None),  # Use get to avoid KeyError
            "date_created": entry["date_created"],
            "date_edited": entry["date_edited"],
            "keywords": ", ".join(entry["keywords"]),
            "dominant_emotion": entry["sentiment"]["label"],
            "dominant_score": entry["sentiment"]["score"], 
        }
        
        # Add all emotions
        for emotion in entry["sentiment"]["emotions"]:
            flat_entry[f"emotion_{emotion['emotion']}"] = emotion["score"]
        
        flattened_entries.append(flat_entry)
    
    df = pd.DataFrame(flattened_entries)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Data saved to {filename}")

def save_to_json(entries, filename="analyzed_journal_entries.json"):
    """Save entries to JSON with flattened sentiment structure."""
    flattened_entries = []
    
    for entry in entries:
        # Check for missing ID
        if "journal_id" not in entry:
            print("Warning: Missing ID in entry:", entry)  # Debugging line
        
        flat_entry = {
            "title": entry["title"],
            "journal_id": entry.get("journal_id", None),  # Use get to avoid KeyError
            "date_created": entry["date_created"],
            "date_edited": entry["date_edited"],
            "keywords": entry["keywords"],
            "emotions": {
                "dominant": {
                    "name": entry["sentiment"]["label"],
                    "score": entry["sentiment"]["score"]
                },
                "all": [
                    {
                        "name": e["emotion"],
                        "score": e["score"]
                    } for e in entry["sentiment"]["emotions"]
                ]
            }
        }
        flattened_entries.append(flat_entry)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(flattened_entries, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {filename}")

def save_to_sqlite(entries, db_filename="journal_data.db"):
    """Save entries to SQLite with emotions in separate table."""
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            journal_id TEXT,
            date_created TEXT,
            date_edited TEXT,
            keywords TEXT,
            dominant_emotion TEXT,
            dominant_score FLOAT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entry_emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER,
            emotion TEXT,
            score FLOAT,
            FOREIGN KEY (entry_id) REFERENCES journal_entries (id)
        )
    """)
    
    # Insert data
    for entry in entries:
        # Check for missing ID
        if "journal_id" not in entry:
            print("Warning: Missing ID in entry:", entry)  # Debugging line
        
        # Debugging: Print entry before saving
        # print(f"Saving entry: {entry}")  # Check the values being saved
        
        # Insert main entry
        try:
            cursor.execute("""
                INSERT INTO journal_entries (
                    title, journal_id, date_created, date_edited,
                    keywords, dominant_emotion, dominant_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entry["title"],
                entry.get("journal_id", None),  # Use get to avoid KeyError
                entry["date_created"],
                entry["date_edited"],
                ", ".join(entry["keywords"]),
                entry["sentiment"]["label"],
                entry["sentiment"]["score"]
            ))
            
            entry_id = cursor.lastrowid
            
            # Insert all emotions
            for emotion in entry["sentiment"]["emotions"]:
                cursor.execute("""
                    INSERT INTO entry_emotions (entry_id, emotion, score)
                    VALUES (?, ?, ?)
                """, (entry_id, emotion["emotion"], emotion["score"]))
        except Exception as e:
            print(f"Error inserting entry: {e}")     

    conn.commit()
    conn.close()
    print(f"Data saved to {db_filename}")

# List all databases the integration can access
#databases = notion.search(filter={"property": "object", "value": "database"})
#print(f"Notion Databases: {databases}\n")


# ------------------ Robust Notion Call to handle API rate limit ------------------
# Global list to collect error messages (or failed page IDs)
error_log = []

def robust_notion_call(api_call, *args, **kwargs):
    """
    Calls an API function with retries on rate-limit errors.
    Records errors in error_log if max attempts are reached.
    """
    max_attempts = 5
    delay = 60  # start with a 60-second delay
    for attempt in range(max_attempts):
        try:
            return api_call(*args, **kwargs)
        except Exception as e:
            if "rate limited" in str(e).lower():
                print(f"Rate limited on attempt {attempt+1}/{max_attempts}. Sleeping for {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                error_log.append(f"Error in {api_call.__name__} with args {args}: {e}")
                raise e
    # If we exhausted all attempts, record the error and raise.
    error_log.append(f"Max attempts reached for {api_call.__name__} with args {args}")
    raise Exception("Max attempts reached for API call.")

# ------------------ Fetch Page Content ------------------
def fetch_page_content(page_id):
    """
    Fetches full content of a Notion page by recursively retrieving all blocks.
    Returns a tuple: (full_content, child_page_ids)
    where full_content is the concatenated content for this page (excluding child pages)
    and child_page_ids is a list of IDs for any child pages encountered.
    """
    try:
        skipped_types = set()
        page = notion.pages.retrieve(page_id)
        title_prop = page["properties"].get("Name", {}).get("title", [])
        page_title = title_prop[0]["plain_text"] if title_prop else "Untitled"

        child_pages = []  # List to store child page IDs

        def get_block_content(block_id):
            blocks = notion.blocks.children.list(block_id=block_id)
            content = []
            for block in blocks.get("results", []):
                block_type = block["type"]

                if block_type in [
                    "paragraph", "heading_1", "heading_2", "heading_3",
                    "bulleted_list_item", "numbered_list_item", "quote",
                    "callout", "toggle", "to_do"
                ]:
                    rich_text = block[block_type].get("rich_text", [])
                    if rich_text:
                        text = "".join([t["plain_text"] for t in rich_text])
                        if block_type.startswith("heading_"):
                            level = block_type[-1]
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
                        else:
                            content.append(text)
                        # Process nested blocks if any
                        if block.get("has_children", False):
                            nested_content, nested_child_pages = get_block_content(block["id"])
                            content.extend("  " + line for line in nested_content.split("\n") if line)
                            # Append any nested child pages to our list
                            child_pages.extend(nested_child_pages)

                # Instead of concatenating child page content, record its ID
                elif block_type == "child_page":
                    child_page_id = block["id"]
                    child_pages.append(child_page_id)
                    # Optionally, you could include a placeholder like:
                    content.append(f"[Child page: {block['child_page']['title']}]")
                
                elif block_type == "table":
                    table_rows = []
                    table_block_id = block["id"]
                    table_children = notion.blocks.children.list(block_id=table_block_id)
                    for row_block in table_children.get("results", []):
                        if row_block["type"] == "table_row":
                            row_cells = []
                            for cell in row_block["table_row"]["cells"]:
                                cell_text = "".join([t["plain_text"] for t in cell]) if cell else ""
                                row_cells.append(cell_text)
                            table_rows.append(row_cells)
                    if table_rows:
                        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table_rows)]
                        header_row = table_rows[0]
                        content.append("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
                        content.append("|" + "|".join(f" {cell:<{width}} " for cell, width in zip(header_row, col_widths)) + "|")
                        content.append("+" + "+".join("=" * (width + 2) for width in col_widths) + "+")
                        for row in table_rows[1:]:
                            content.append("|" + "|".join(f" {str(cell):<{width}} " for cell, width in zip(row, col_widths)) + "|")
                            content.append("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
                        content.append("")
                else:
                    if block_type not in skipped_types:
                        skipped_types.add(block_type)
            return "\n".join(content), child_pages

        full_content, child_pages_local = get_block_content(page_id)
        return full_content, child_pages

    except Exception as e:
        # Use page_id as fallback if title not available
        print(f"🚨 ERROR fetching page content for page_id '{page_id}': {e}")
        return "", []



# ------------------ Fetch Journal Entries ------------------
"""
Fetch journal entries from the Notion database.
If n is provided, only the most recent n entries (including child pages) are loaded.
Child pages encountered within a parent page are treated as separate journal entries.
In your fetch functions, when an error occurs (e.g. processing a child page), record it:
"""
def fetch_journal_entries(database_id, n=None):
    processed_child_pages = set()  # track subpages we've already processed
    entries = []
    has_more = True
    next_cursor = None
    sorts = [{"timestamp": "last_edited_time", "direction": "descending"}] if n else None
    start_time = time.time()

    with tqdm(desc="Fetching Entries", unit="entry") as pbar:
        while has_more:
            params = {"page_size": 100}
            if next_cursor:
                params["start_cursor"] = next_cursor
            if sorts:
                params["sorts"] = sorts

            try:
                response = robust_notion_call(notion.databases.query, database_id=database_id, **params)
            except Exception as e:
                print(f"❌ ERROR: Notion API request failed: {e}")
                break

            for result in response.get("results", []):
                parent_page_id = result["id"]
                try:
                    parent_content, child_page_ids = fetch_page_content(parent_page_id)
                except Exception as e:
                    error_log.append(f"Failed to fetch content for page {parent_page_id}: {e}")
                    parent_content = ""
                    child_page_ids = []

                # Get parent's title from the DB item
                properties = result["properties"]
                title_prop = properties.get("Name", {}).get("title", [])
                if title_prop:
                    parent_title = title_prop[0].get("plain_text", "Untitled")
                else:
                    parent_title = "Untitled"

                date_created = result.get("created_time", "")
                date_edited = result.get("last_edited_time", "")
                entries.append({
                    "title": parent_title,
                    "content": parent_content,
                    "date_created": date_created,
                    "date_edited": date_edited,
                    "journal_id": parent_page_id,
                })
                pbar.update(1)

                # Process each child page exactly once
                for child_page_id in child_page_ids:
                    if child_page_id in processed_child_pages:
                        # Already processed
                        continue
                    processed_child_pages.add(child_page_id)

                    try:
                        child_page = robust_notion_call(notion.pages.retrieve, child_page_id)
                        # Attempt "Name" first (common default in many Notion databases)
                        title_prop = child_page.get("properties", {}).get("Name", {}).get("title", [])

                        # If that didn't work, try "title" property
                        if not title_prop:
                            title_prop = child_page.get("properties", {}).get("title", {}).get("title", [])

                        # If that still fails, check if it's a child_page
                        if title_prop:
                            child_title = title_prop[0].get("plain_text", "Untitled")
                        elif "child_page" in child_page:
                            child_title = child_page["child_page"]["title"]
                        else:
                            child_title = "Untitled"

                        child_date_created = child_page.get("created_time", "")
                        child_date_edited = child_page.get("last_edited_time", "")
                        child_content, _ = fetch_page_content(child_page_id)

                        #print("Found child page title:", child_title)
                        entries.append({
                            "title": child_title,
                            "content": child_content,
                            "date_created": child_date_created,
                            "date_edited": child_date_edited,
                            "journal_id": child_page_id,
                        })
                        pbar.update(1)
                    except Exception as e:
                        error_log.append(f"Failed processing child page {child_page_id}: {e}")
                        print(f"❌ ERROR processing child page {child_page_id}: {e}")

                if n is not None and len(entries) >= n:
                    has_more = False
                    break

            has_more = response.get("has_more", False)
            next_cursor = response.get("next_cursor", None)
            elapsed_time = time.time() - start_time
            avg_time_per_entry = elapsed_time / max(len(entries), 1)
            estimated_total_time = avg_time_per_entry * max(len(entries), 1)
            remaining_time = max(0, estimated_total_time - elapsed_time)
            pbar.set_postfix({
                "Elapsed": f"{elapsed_time:.1f}s",
                "ETA": f"{remaining_time:.1f}s"
            })

    print(f"\nFetch completed in {time.time() - start_time:.2f} seconds\n")
    if n is not None:
        entries = entries[:n]
    return entries



# ------------------ Sentiment & Keyword Analysis ------------------

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
        return {
            "label": "neutral",
            "score": 0.0,
            "emotions": [
                {"emotion": "neutral", "score": 0.0}
            ]
        }

    chunks = textwrap.wrap(text, chunk_size)
    all_emotions = []  # Initialize the list here

    for chunk in chunks:
        results = sentiment_model(chunk)[0]  # Get all emotion scores
        all_emotions.append(results)  # Store all emotions for this chunk

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

    # Average the scores
    for label in final_emotions:
        final_emotions[label] /= len(chunks)

    # Format the return value to include all emotions
    return {
        "label": max(final_emotions.items(), key=lambda x: x[1])[0],  # Dominant emotion
        "score": max(final_emotions.items(), key=lambda x: x[1])[1],  # Score of dominant emotion
        "emotions": [
            {"emotion": e, "score": s} 
            for e, s in sorted(final_emotions.items(), key=lambda x: x[1], reverse=True)
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

    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)  # ✅ Use 'english' instead of custom stopwords
    try:
        
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        keywords = feature_names[:10] if len(feature_names) > 0 else ["NO_KEYWORDS_FOUND"]# Return top 10 words
        return list(keywords)
    except ValueError as e:
        print(f"🚨 ERROR: Could not process text: {repr(text)}")
        return []


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
                "journal_id": entry.get("journal_id", None),
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


# ------------------ Main Execution ------------------

# Fetch all journal entries
entries = fetch_journal_entries(NOTION_DATABASE_ID)
print(f"Total entries fetched: {len(entries)}")

# Print out a summary of errors (if any)
if error_log:
    print("\nSummary of errors encountered during fetching:")
    for err in error_log:
        print(err)
else:
    print("\nNo errors encountered during fetching.")

# Clean the data
cleaned_entries = clean_entries(entries)

# Analyze sentiment & keywords
analyzed_entries = analyze_journal_entries(cleaned_entries)

# Save the data to different formats
save_to_csv(analyzed_entries)
save_to_json(analyzed_entries)
save_to_sqlite(analyzed_entries)

print (f"Total entries: {len(entries)}")