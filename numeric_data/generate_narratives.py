import pandas as pd

cancer_types = ['colorectal', 'gastric', 'pancreatic.csv']
train_narratives = []
test_narratives = []
test_cancer_types = []

for cancer_type in cancer_types:
    dataset = pd.read_csv(f'{cancer_type}_cancer_dataset.csv')
    
    """
    1) Randomly select 100 rows from the dataset.
    2) Let the first 80 rows be for the training set and the last 20 rows be for the test set.
    3) For each row in the training set, generate a 2-3 sentence narrative using a random patient name that captures most of the
       information in the row. You'll be given the column names for each feature. The narrative should be a
       brief summary of the patient based on the features in the row. At the end of the narrative, include
       what type of cancer the patient has (and the stage if applicable).
    4) For each row in the test set, generate the same narrative but leave out the type of cancer (and stage) at the end.
       Instead, save the cancery type in a separate data structure.
    5) Aggregate the training and test narratives across all cancer types.
    6) For the train set, save the narratives with a line of separation between each narrative in a text file.
    7) For the test set, save the narratives and the separate cancer type in a CSV file.
    """
    
    # Randomly select 100 rows from the dataset
    selected_rows = dataset.sample(n=100, random_state=42)
    
    # Split into train and test sets
    train_set = selected_rows.iloc[:80]
    test_set = selected_rows.iloc[80:]
    
    for index, row in train_set.iterrows():
        # Generate a JSON of the column names to their values so we can pass it to the LLM
        patient_data = row.to_dict()
        # stringify the JSON
        patient_data_str = str(patient_data)
        
        
    
    