# Annotation Essay Arguments using OpenAI API

Data for processing is in `./processed_asap_aes_data.tsv`.
Columns used are `essay_id`, `essay_set`, and `essay`.
Essay prompts can be found in `../essay_scoring/asap-aes/prompts/`.

# Running the code

Run the following to 
```
export OPENAI_API_KEY=<YOUR KEY HERE>

python segment_essays.py --essays <ESSAY_DATA> --prompts <PROMPTS_PATH> --output <OUTPUT_DIR> --task <TASK>
```

