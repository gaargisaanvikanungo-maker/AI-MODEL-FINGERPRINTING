import os
import time
import pandas as pd
from google import genai
from google.genai.errors import APIError

# 1. Client Configuration
# REPLACE THIS WITH YOUR ACTUAL GEMINI API KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

INPUT_EXCEL_PATH = "data/dataset.csv.xlsx"
OUTPUT_CSV_PATH = "data/dataset_clean_gemini.csv"
COLUMN_NAME = 'GEMINI 2.5 PRO'

# 2. Load Dataset Safely
if os.path.exists(OUTPUT_CSV_PATH):
    print("Found progress file. Resuming and checking for missing or error rows...")
    df = pd.read_csv(OUTPUT_CSV_PATH)
elif os.path.exists(INPUT_EXCEL_PATH):
    print(f"Starting fresh from your clean Excel file: {INPUT_EXCEL_PATH}")
    df = pd.read_excel(INPUT_EXCEL_PATH)
else:
    raise FileNotFoundError(f"Could not find the file at {INPUT_EXCEL_PATH}")

# Ensure target column exists
if COLUMN_NAME not in df.columns:
    df[COLUMN_NAME] = None

# Drop trailing empty spreadsheet rows
df = df.dropna(subset=['prompt_text'])
print(f"Total rows in system: {len(df)}")

# 3. Data Collection Loop
for index, row in df.iterrows():
    # Clear out previous error text so it retries failed prompts, but skip successful ones
    current_val = str(row.get(COLUMN_NAME)).strip()
    if pd.notna(row.get(COLUMN_NAME)) and current_val not in ["", "nan", "None", "ERROR: API Issue", "ERROR: System Failure"]:
        continue

    prompt = row['prompt_text']
    print(f"Processing prompt {index + 1}/{len(df)}: {str(prompt)[:50]}...")

    success = False
    while not success:
        try:
            # Query using the stable, free flash model to bypass the pro paywall
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt
            )

            # Map response text back into your exact Excel structure column
            df.at[index, COLUMN_NAME] = response.text
            success = True
            print(f"  ✅ Saved response for prompt {index + 1}")

            # Save instantly to disk so you never lose progress
            df.to_csv(OUTPUT_CSV_PATH, index=False)

            # Anti-rate-limit buffer: 8 seconds prevents running out of Tokens Per Minute (TPM)
            time.sleep(8)

        except APIError as e:
            error_str = str(e).upper()

            # Error Case A: Server Overloaded (503 / UNAVAILABLE) -> Wait 30 seconds and retry row
            if "503" in error_str or "UNAVAILABLE" in error_str:
                print(
                    "  ⚠️ Google servers are busy (503). Waiting 30 seconds to retry this prompt...")
                time.sleep(30)

            # Error Case B: Quota Limit Crossed (429) -> Wait 75 seconds for rolling window reset
            elif "429" in error_str or "QUOTA" in error_str:
                print(
                    "  ⚠️ Speed limit reached! Pausing for 75 seconds to fully clear window...")
                time.sleep(75)

            # Error Case C: Any other unexpected API structural breakdown -> Skip to next row
            else:
                print(f"  ❌ Permanent API Error at prompt {index + 1}: {e}")
                df.at[index, COLUMN_NAME] = "ERROR: API Issue"
                df.to_csv(OUTPUT_CSV_PATH, index=False)
                success = True
                time.sleep(8)

        except Exception as e:
            print(f"  ❌ Unexpected system failure at prompt {index + 1}: {e}")
            df.at[index, COLUMN_NAME] = "ERROR: System Failure"
            df.to_
