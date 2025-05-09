"""
Go through each row in 'test_narratives.csv' and check if the narrative does not contain any of
['colorectal', 'gastric', 'pancreatic'].

If it does, print out the narrative.
"""

import pandas as pd
import os

df = pd.read_csv('test_narratives.csv')
print('Number of test narratives:', len(df))

print('Narratives that contain cancer types:')
for index, row in df.iterrows():
    narrative = row['Narrative']
    if any(cancer_type in narrative.lower() for cancer_type in ['colorectal', 'gastric', 'pancreatic']):
        print(f"Index: {index}, Narrative: {narrative}")
        print("~"*50)
        
