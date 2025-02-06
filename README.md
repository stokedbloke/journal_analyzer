# Journal Analyzer

A Python tool that integrates with Notion to analyze journal entries using sentiment analysis and natural language processing.

## Features
- Fetches journal entries from a Notion database
- Processes various Notion block types:
  - Text blocks (paragraphs, headings)
  - Lists (bulleted and numbered)
  - Tables
  - Quotes and callouts
- Performs sentiment analysis on entries
- Extracts keywords from content
- Caches results for efficiency

## Prerequisites
- Python 3.12+
- Notion API access token
- A Notion database containing journal entries

## Installation

1. Clone the repository:

bash
git clone [your-repo-url]
cd journal_analyzer

2. Create a virtual environment:

bash
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate


3. Install dependencies:

bash
pip install -r requirements.txt

4. Create a `journal_analyzer.env` file with your Notion credentials:

env
NOTION_API_TOKEN=your_token_here
NOTION_DATABASE_ID=your_database_id_here


## Usage
Run the analyzer:

bash
python journal_analyzer.py

## Configuration
- Modify sentiment analysis model size in the code
- Adjust keyword extraction parameters
- Configure caching behavior

## Project Structure
- `journal_analyzer.py`: Main script for fetching and analyzing journal entries
- `requirements.txt`: List of dependencies
- `journal_analyzer.env`: Environment variables for Notion API credentials (not included in repo)
- `.gitignore`: Git ignore rule

## Dependencies
- `notion-client`: Notion API integration
- `transformers`: Sentiment analysis
- `nltk`: Natural language processing
- `pandas`: Data handling
- `scikit-learn`: Keyword extraction
- `python-dotenv`: Environment variable management
- `tqdm`: Progress bars