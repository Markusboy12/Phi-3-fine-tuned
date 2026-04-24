# dataset formatting

import pandas as pd
import json
from datasets import Dataset
from unsloth import FastLanguageModel

with open('healthcare_data2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


records = []
for item in data:
    user_text = item['prompt'][0]['content'] if isinstance(item['prompt'], list) else item['prompt']['content']
    assistant_text = item['completion'][0]['content'] if isinstance(item['completion'], list) else item['completion']['content']

    records.append({
        "user": user_text,
        "assistant": assistant_text
    })

df = pd.DataFrame(records)

def formatting_prompts_func(examples):
    convos = []
    for user, assistant in zip(examples['user'], examples['assistant']):
        text = f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
        convos.append(text)
    return {"text": convos}


dataset = Dataset.from_pandas(df)
dataset = dataset.map(formatting_prompts_func, batched=True)

print(dataset[0]['text'][:200])