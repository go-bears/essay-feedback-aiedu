import os
import csv
import time
import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures
import json

# Load environment variables (for API key)
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# File paths
INPUT_CSV = "sampling-25.csv"
OUTPUT_CSV = "argument_components_results.csv"
BATCH_SIZE = 3  # Number of texts to process in a single batch
MAX_WORKERS = 3  # Number of concurrent batches to process

# Define the prompt template
PROMPT_TEMPLATE = """
# TASK
Segment the following essay into distinct argument components: 
- <Lead>: An introduction that begins with a statistic, quotation, description, or other device to grab the reader's attention and point toward the thesis. 
- <Position>: An opinion or conclusion on the main question. 
- <Claim>: A statement that supports the position. 
- <Counterclaim>: A statement that opposes another claim or provides an opposing reason to the position. 
- <Rebuttal>: A statement that refutes a counterclaim. 
- <Evidence>: Ideas or examples that support claims, counterclaims, or rebuttals. 
- <Concluding Statement>: A statement that restates the claims and summarizes the argument.

After each argument component, insert the corresponding marker, e.g. insert <Lead> after a lead component. Keep the original text in the same order without adding, removing, or altering any words (other than inserting the markers). Do not correct for spelling, grammar, or punctuation errors in the original text.

# GUIDELINES
Identify each coherent segment that forms a logical unit of the argument (e.g., claims, premises, evidence, or conclusions). If there are multiple claims, counterclaims, or rebuttals, identify each as a separate component with an enumeration added to the marker. Make sure that the markers are not nested.

Text to analyze:
{text}
"""

def process_batch(batch_texts):
    """Process a batch of text entries through the Anthropic API"""
    try:
        # Create a list to store the results
        results = []
        
        # Process each text in the batch
        for text_data in batch_texts:
            text_id = text_data['id']
            text = text_data['text']
            
            try:
                # Call Anthropic API
                response = client.messages.create(
                    model="claude-3-7-sonnet-20250219",  # You can adjust the model as needed
                    max_tokens=2000,
                    temperature=0,
                    system="""You are an expert at identifying essay argument components in text.

                    Please parse the provided text for the components of an argument in an essay. The argument types are as follows: 
                    - Lead: An introduction that begins with a statistic, quotation, description, or other device to grab the reader's attention and point toward the thesis. 
                    - Position: An opinion or conclusion on the main question. 
                    - Claim: A statement that supports the position. 
                    - Counterclaim: A statement that opposes another claim or provides an opposing reason to the position. 
                    - Rebuttal: A statement that refutes a counterclaim. 
                    - Evidence: Ideas or examples that support claims, counterclaims, or rebuttals. 
                    - Concluding Statement: A statement that restates the claims and summarizes the argument
                    """,
                    messages=[
                        {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
                    ]
                )
                
                # Add the result to the list
                results.append({
                    'original_id': text_id,
                    'original_text': text,
                    'processed_text': response.content[0].text
                })
                
                # Add a small delay between API calls within a batch
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error processing text {text_id}: {e}")
                results.append({
                    'original_id': text_id,
                    'original_text': text,
                    'processed_text': f"ERROR: {str(e)}"
                })
        
        return results
    
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []

def main():
    # Check if API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your API key in a .env file or environment variable")
        return
    
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found")
        return
    
    # Create a list to store the texts
    texts = []
    
    try:
        # Read the input CSV
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded CSV with columns: {df.columns}")
        print(f"Number of rows: {len(df)}")
        
        # Check if the CSV has an essay-id column
        if 'essay-id' in df.columns:
            print("Found 'essay-id' column, will process text files from the sampling-25 directory")
            
            # Process each file
            for idx, filename in enumerate(df["essay-id"]):
                if filename.endswith(".txt"):
                    try:
                        file_path = os.path.join("sampling-25", filename)
                        if os.path.exists(file_path):
                            with open(file_path, "r", encoding="utf-8") as f:
                                text = f.read()
                                texts.append({
                                    'id': filename,
                                    'text': text
                                })
                        else:
                            print(f"Warning: File not found: {file_path}")
                    except Exception as e:
                        print(f"Error reading file {filename}: {e}")
            
        print(f"Loaded {len(texts)} text files")
                    
        # Create batches of texts
        batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
        print(f"Created {len(batches)} batches of size {BATCH_SIZE}")
        
        # Create a list to store all results
        all_results = []
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all batches for processing
            future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    # Save intermediate results
                    pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
                    print(f"Saved progress (batch {batch_idx + 1}/{len(batches)}, total texts: {len(all_results)}/{len(texts)})")
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
        
        print(f"Processing complete. Results saved to {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 