import pandas as pd
import random
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import csv

load_dotenv()

client = OpenAI(api_key=os.getenv("GRAPHRAG_API_KEY"))

# List of possible patient names for randomization
first_names = ["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "William", "Linda", 
               "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", 
               "Charles", "Karen", "Daniel", "Nancy", "Matthew", "Lisa", "Anthony", "Margaret"]
last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", 
              "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", 
              "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee"]

cancer_types = ['colorectal', 'gastric', 'pancreatic']
train_narratives = []
test_narratives = []
test_cancer_types = []

def generate_narrative(patient_data, include_cancer_type=True, cancer_type=None):
    patient_name = f"{random.choice(first_names)} {random.choice(last_names)}"
    
    if include_cancer_type:
        prompt = f"""Generate a 2-3 sentence narrative about a patient named {patient_name} based on the following medical data. 
        Include their cancer type ({cancer_type}) and stage (if applicable) at the end of the narrative.
        Medical data: {patient_data}"""
    else:
        prompt = f"""Generate a 2-3 sentence narrative about a patient named {patient_name} based on the following medical data. 
        DO NOT mention their cancer type or stage in the narrative.
        Medical data: {patient_data}"""
    
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a medical narrative generator that creates brief patient summaries based on medical data. Just output the narratives as specified and nothing else."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )
    narrative = response.choices[0].message.content.strip()
    return narrative

for cancer_type in cancer_types:
    file_path = f'{cancer_type}_cancer_dataset.csv'
    dataset = pd.read_csv(file_path)

    selected_rows = dataset.sample(n=100, random_state=42)
    
    train_size = 80
    test_size = 20
    
    train_set = selected_rows.iloc[:train_size]
    test_set = selected_rows.iloc[train_size:]
    
    print(f"Processing {cancer_type} cancer dataset: {train_size} training samples, {test_size} test samples")
    
    # Process training set
    for index, row in train_set.iterrows():
        patient_data = row.to_dict()

        narrative = generate_narrative(patient_data, include_cancer_type=True, cancer_type=cancer_type)
        train_narratives.append(narrative)
    
    # Process test set
    for index, row in test_set.iterrows():
        patient_data = row.to_dict()

        narrative = generate_narrative(patient_data, include_cancer_type=False)
        test_narratives.append(narrative)

        test_cancer_types.append(cancer_type)
    
assert len(train_narratives) == 80 * len(cancer_types), "Training narratives count mismatch"

with open('../input/train_narratives_colorectal.txt', 'w') as f:
    for narrative in train_narratives[:80]:
        f.write(narrative + '\n\n')

with open('../input/train_narratives_gastric.txt', 'w') as f:
    for narrative in train_narratives[80:160]:
        f.write(narrative + '\n\n')

with open('../input/train_narratives_pancreatic.txt', 'w') as f:
    for narrative in train_narratives[160:]:
        f.write(narrative + '\n\n')

with open('test_narratives.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Narrative', 'Cancer Type'])
    for narrative, cancer_type in zip(test_narratives, test_cancer_types):
        writer.writerow([narrative, cancer_type])

print(f"Generated {len(train_narratives)} training narratives and {len(test_narratives)} test narratives")
print(f"Training narratives saved to train_narratives.txt")
print(f"Test narratives saved to test_narratives.csv")
