# Argument Component Extraction

This repo contains a python module (`essay_feedback`) that processes text samples from a CSV file, sends them to either the Anthropic API or OpenAI API to identify argument components, and saves the results to a new CSV file.

## Setup

1. Install the package:
   ```
   pip install -e essay_feedback
   ```

2. Create a `.env` file in the same directory as your script (not necessarily the package!) with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   OPENAI_API_KEY=your_other_api_key_here
   ```

3. Import the package in your script
   ```python
   import essay_feedback
   from essay_feedback.data import *
   ```

<!-- 4. Make sure your input CSV file (`sampling-25.csv`) is in the same directory as the script. -->

## Usage

Run the script with:
```
python process_samples.py
```

The script will:
1. Read text samples from `sampling-25.csv`
2. Automatically identify which column contains the text data
3. Process each text through the Anthropic API to identify argument components
4. Save the results to `argument_components_results.csv`

## Output

The output CSV file will contain:
- `original_id`: The ID from the original CSV (or row index if no ID column)
- `original_text`: The original text from the input CSV
- `processed_text`: The text with argument components marked using markdown tags:
  - Claims: `**[CLAIM]** text **[/CLAIM]**`
  - Premises: `**[PREMISE]** text **[/PREMISE]**`
  - Major claims: `**[MAJOR_CLAIM]** text **[/MAJOR_CLAIM]**`

## Notes

- The script includes error handling and progress tracking
- Results are saved incrementally (every 5 entries) to prevent data loss
- A small delay is added between API calls to avoid rate limiting 
>>>>>>> 3c0b88e (initial commit processing texts)
