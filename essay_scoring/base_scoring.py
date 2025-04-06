import os
import csv
import time
import ollama
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import json
from datetime import datetime

import essay_feedback
from essay_feedback.data import *

output_dir = "base-scoring-outputs"
os.makedirs(output_dir, exist_ok=True)
date_str = datetime.now().strftime("%Y-%m-%d")
model = "llama3.1"
output_file = os.path.join(output_dir, f"{model}-scoring-output-{date_str}.csv")


model = LLM(model=model)

prompts_rubrics = {}

prompts = os.listdir(os.path.join(essay_scoring.asap_aes, "prompts"))
rubrics = os.listdir(os.path.join(essay_scoring.asap_aes, "rubrics"))

print(prompts)
print(rubrics)








