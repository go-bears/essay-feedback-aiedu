import csv
import re
import datetime

DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

def process_essay(processed_text):
    """
    Process the processed_text by extracting all tagged segments.
    For each segment, compute the absolute word indices for the words inside that tag,
    counting continuously from the start of the processed_text (i.e. numbering does not restart).
    
    Returns a list of tuples (tag, indices) where indices is a list of absolute word indices.
    """
    # Pattern to capture tags and their contents: e.g. <Claim3> ... </Claim3>
    pattern = re.compile(r'<([^>]+)>(.*?)</\1>', re.DOTALL)
    
    segments = []
    cumulative = 0  # Running total of words seen so far.
    for match in pattern.finditer(processed_text):
        # Extract the tag name and remove any digits from it.
        tag = match.group(1).strip()
        tag = re.sub(r'\d+', '', tag)
        text = match.group(2).strip()
        words = text.split()  # Split the segment text by whitespace.
        # Compute indices for this segment, starting from cumulative .
        indices = list(range(cumulative, cumulative + len(words)))
        cumulative += len(words)
        segments.append((tag, indices))
    return segments

def main():
    input_file = 'argument_components_results_2025-03-12_13-34.csv'
    output_file = f'results_{DATETIME}.csv'
    
    with open(input_file, newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        
        # Write header row.
        writer.writerow(['id', 'class', 'predictionstring'])
        
        # Process each essay in the input file.
        for row in reader:
            original_id = row.get('original_id', '').strip()
            # Remove the .txt extension if present.
            essay_id = original_id.rsplit('.', 1)[0]
            processed_text = row.get('processed_text', '')
            
            # Process the text to extract segments and their word indices.
            segments = process_essay(processed_text)
            for tag, indices in segments:
                predictionstring = " ".join(map(str, indices))
                writer.writerow([essay_id, tag, predictionstring])
    
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()