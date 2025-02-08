from dotenv import load_dotenv
import os

print(f"Loading environment from: journal_analyzer.env")
load_dotenv("journal_analyzer.env")

OURA_API_TOKEN = os.getenv("OURA_API_TOKEN")
V2_SLEEP_URL = 'https://api.ouraring.com/v2/usercollection/sleep'

import requests
import datetime
import sqlite3

def fetch_sleep_data(start_date, end_date):
    headers = {'Authorization': f'Bearer {OURA_API_TOKEN}'}
    params = {
        'start_date': start_date,  # Format: YYYY-MM-DD
        'end_date': end_date       # Format: YYYY-MM-DD
    }
    
    response = requests.get(V2_SLEEP_URL, headers=headers, params=params)
    
    # Debug: print the URL and full response for troubleshooting.
    #print("Request URL:", response.url)
    if response.status_code == 200:
        json_data = response.json()
        #print("Full JSON Response:", json.dumps(json_data, indent=2))  # Debug: Check for 'score' field
        return json_data.get('data', [])  # ✅ Extract sleep data
    else:
        print("Error fetching sleep data:", response.text)
        return []
    

def create_tables(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS OuraSleep (
            date TEXT PRIMARY KEY,
            temperature_deviation REAL,
            total_sleep REAL,
            efficiency REAL,
            hrv REAL,
            respiratory_rate REAL,
            average_heart_rate REAL
        )
    ''')
    conn.commit()
    conn.close()

import json
import pandas as pd

def output_oura_data(oura_data, csv_filename='oura_data.csv', json_filename='oura_data.json'):
    """
    Outputs Oura data to CSV and JSON files.

    Parameters:
    - oura_data: List of dictionaries containing Oura data.
    - csv_filename: Name of the CSV file to save the data.
    - json_filename: Name of the JSON file to save the data.
    """
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(oura_data)

    # Output to CSV
    df.to_csv(csv_filename, index=False)
    print(f"Oura data saved to {csv_filename}")

    # Output to JSON
    with open(json_filename, 'w') as json_file:
        json.dump(oura_data, json_file, indent=4)
    print(f"Oura data saved to {json_filename}")


import json
import sqlite3

def insert_sleep_data(db_path, sleep_entries):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    for entry in sleep_entries:
        if entry is None:
            print("Skipping None entry")
            continue  # Skip this iteration if entry is None
        
        date = entry.get('day')  
        if not date:  
            print("Skipping entry with missing date:", entry)
            continue  # Skip if date is still missing
        
        # ✅ Fix: Extract sleep score 
        #sleep_score = entry.get('score') or entry.get('sleep_score')

        # ✅ Fix: Extract total sleep in seconds
        total_sleep = entry.get('total_sleep_duration')

        efficiency = entry.get('efficiency')

        # ✅ Fix: Ensure readiness is a dictionary before accessing keys
        readiness = entry.get('readiness')
        if isinstance(readiness, dict):  
            temperature_deviation = readiness.get('temperature_deviation', None)  
        else:
            temperature_deviation = None  # Assign None if readiness is missing

        # ✅ Fix: Ensure HRV, respiratory rate, and average heart rate are correctly extracted
        hrv = entry.get('average_hrv')  # Correct key for HRV
        respiratory_rate = entry.get('average_breath')  # Correct key for Respiratory Rate
        average_heart_rate = entry.get('average_heart_rate')  # Correct key for Average Heart Rate
        
        # Debug print before inserting
        #print(f"Inserting: {date}, Sleep Score: {sleep_score}, Temperature Deviation: {temperature_deviation}, "
        #      f"Total Sleep: {total_sleep}, Efficiency: {efficiency}, HRV: {hrv}, "
        #      f"Respiratory Rate: {respiratory_rate}, Average Heart Rate: {average_heart_rate}")

        try:
            # Insert into the database
            c.execute('''
                INSERT OR REPLACE INTO OuraSleep (date, total_sleep, efficiency, temperature_deviation, hrv, respiratory_rate, average_heart_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (date, total_sleep, efficiency, temperature_deviation, hrv, respiratory_rate, average_heart_rate))
        except Exception as e:
            print(f"Error inserting entry: {e}")
    
    conn.commit()
    conn.close()



if __name__ == "__main__":
    # Path for the SQLite database file.
    db_path = 'oura_analyzer.db'
        
    # Delete the existing database file (optional)
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Deleted old database file.")

    # Create tables if they don't already exist.
    create_tables(db_path)
    
    # Define your date range.
    # For example, to fetch data for the last 1825 days 5 years: (+200 days) 
    # 01/23/2020 was the first day with oura
    end_date_obj = datetime.date.today()
    #start_date_obj = end_date_obj - datetime.timedelta(days=2025)
    # Hard-code start_date_obj to January 23, 2020
    start_date_obj = datetime.datetime(2020, 1, 23)
    start_date = start_date_obj.strftime('%Y-%m-%d')
    end_date = end_date_obj.strftime('%Y-%m-%d')
    
    print(f"Fetching sleep data from {start_date} to {end_date}")
    
    # Fetch sleep data using Oura API v2.
    sleep_entries = fetch_sleep_data(start_date, end_date)
    output_oura_data(sleep_entries)
    print("Total sleep entries retrieved:", len(sleep_entries))

    # Insert the retrieved sleep data into the database.
    if sleep_entries:
        insert_sleep_data(db_path, sleep_entries)
        print("Sleep data successfully inserted into the database.")
    else:
        print("No sleep data was retrieved; nothing to insert.")