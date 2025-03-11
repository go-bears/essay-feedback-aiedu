import os
import csv
import time
import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures
import json
from datetime import datetime
from examples import examples

# Load environment variables (for API key)
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# File paths
INPUT_CSV = "test.csv"
INPUT_DIR = "test"
OUTPUT_CSV = f"argument_components_results_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
OUTPUT_DIR = "test_output"
BATCH_SIZE = 5  # Number of texts to process in a single batch
MAX_WORKERS = 3  # Number of concurrent batches to process
EXAMPLES = examples

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
Identify each coherent segment that forms a logical unit of the argument (e.g., claims, premises, evidence, or conclusions). Make sure that the markers are not nested.
# EXAMPLES
Here are some examples of how to identify and mark the argument components. The orginal text and the tagged text are separated by a line of dashes are provided for each example.
{EXAMPLES}

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
                # Call OpenAI API
                response = openai.chat.completions.create(
                    model="gpt-4.5-preview",  # You can adjust the model as needed
                    max_tokens=8000,
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
                        {"role": "user", "content": PROMPT_TEMPLATE.format(text=text, EXAMPLES=EXAMPLES)}
                    ]
                )
                
                # Add the result to the list
                results.append({
                    'original_id': text_id,
                    'original_text': text,
                    'processed_text': response.choices[0].message.content
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
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
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
            print(f"Found 'essay-id' column, will process text files from the {INPUT_DIR} directory")
            
            # Process each file
            for idx, filename in enumerate(df["essay-id"]):
                print(f"Processing file {filename}")
                if filename.endswith(".txt"):
                    try:
                        file_path = os.path.join(INPUT_DIR, filename)
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
                    output_file = os.path.join(OUTPUT_DIR, f"openai_batch_{batch_idx + 1}.csv")
                    pd.DataFrame(all_results).to_csv(output_file, index=False)
                    print(f"Saved progress (batch {batch_idx + 1}/{len(batches)}, total texts: {len(all_results)}/{len(texts)})")
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
        
        print(f"Processing complete. Results saved to {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 