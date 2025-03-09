import os
import csv
import time
import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import shutil


# Load environment variables (for API key)
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# File paths
INPUT_CSV = "./sampling-25.csv"
OUTPUT_CSV = "argument_components_results.csv"

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

def process_text(text):
    """Process a single text entry through the Anthropic API"""
    try:
        # Add delay to avoid rate limiting
        time.sleep(0.5)
        
        # Call Anthropic API
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",  # You can change the model as needed
            max_tokens=2000,
            temperature=0,
            system="""You are an expert at identifying essay argument components in text.

            Please parse the provided text for the components of an argument in an essay. The argument types are as follows: - Lead: An introduction that begins with a statistic, quotation, description, or other device to grab the reader’s attention and point toward the thesis. - Position: An opinion or conclusion on the main question. - Claim: A statement that supports the position. - Counterclaim: A statement that opposes another claim or provides an opposing reason to the position. - Rebuttal: A statement that refutes a counterclaim. - Evidence: Ideas or examples that support claims, counterclaims, or rebuttals. - Concluding Statement: A statement that restates the claims and summarizes the argument
            """,
            messages=[
                {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
            ]
        )
        
        # Return the response content
        return response.content[0].text
    except Exception as e:
        print(f"Error processing text: {e}")
        return f"ERROR: {str(e)}"

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
    
    # Read the input CSV
    df = pd.read_csv(INPUT_CSV)
    print(df.columns)
    print(df.head(1))

    texts = []

    try:        
        for filename in df["essay-id"][:3]:
            if filename.endswith(".txt"):
                with open(os.path.join("sampling-25", filename), "r") as f:
                    text = f.read()
                    texts.append((filename, text))
    except Exception as e:
        error = f"An error occurred: {e}"
        texts.append((filename, error))
    
    
    # Process each text entry
    results = []
    for filename, text in tqdm(texts, total=len(texts), desc="Processing texts"):
        try:
            api_response = process_text(text)
            
            # Store the result
            results.append({
                'original_id': filename,
                    'original_text': text,
                    'processed_text': api_response
                })
        except Exception as e:
            print(f"Error processing text: {e}")
            results.append({
                'original_id': filename,
                'original_text': text,
                'processed_text': f"ERROR: {str(e)}"
            })
        # Save intermediate results every 5 entries
        if (len(results) + 1) % 5 == 0 or len(results) == len(texts) - 1:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            print(f"Saved progress ({len(results)}/{len(texts)})")
        
        print(f"Processing complete. Results saved to {OUTPUT_CSV}")
        

if __name__ == "__main__":
    main() 