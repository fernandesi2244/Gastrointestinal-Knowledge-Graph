import os
import random
import csv
from openai import OpenAI
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# Download the punkt tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('punkt_tab')

load_dotenv()

def get_random_chunks(file_path, num_chunks=3):
    """Extract random chunks from a text file, with each chunk being roughly 3 sentences."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Use NLTK's sentence tokenizer for better sentence splitting
        sentences = sent_tokenize(text)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return [text]
        
        chunks = []
        for _ in range(num_chunks):
            if len(sentences) <= 3:
                start_idx = 0
            else:
                start_idx = random.randint(0, max(0, len(sentences) - 3))
            
            chunk = ' '.join(sentences[start_idx:start_idx + 3])
            chunks.append(chunk)
        
        return chunks
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def generate_question(client, chunk):
    """Generate a patient question about the text chunk using GPT-4.1-nano."""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a medical question generator. Given a chunk of medical text, create a single question that a patient might ask their doctor about a fact from that text. The question should be direct and concise. Only respond with the question."},
                {"role": "user", "content": f"Generate a patient question based on this medical text: {chunk}"}
            ],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating question: {e}")
        return "Failed to generate question"

def process_directory(directory, exclude_files, output_file):
    client = OpenAI(api_key=os.getenv("GRAPHRAG_API_KEY"))
    
    files = [f for f in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, f)) 
             and f.endswith('.txt') 
             and f not in exclude_files]
    
    question_chunk_pairs = []
    
    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(directory, filename)
        print(f"Processing {filename}...")
        
        # Extract 3 random chunks from the file
        chunks = get_random_chunks(file_path, num_chunks=3)
        
        for chunk in chunks:
            question = generate_question(client, chunk)
            question_chunk_pairs.append({
                'question': question,
                'text_chunk': chunk
            })
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'text_chunk']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for pair in question_chunk_pairs:
            writer.writerow(pair)
    
    print(f"Completed! Generated {len(question_chunk_pairs)} question-chunk pairs.")

def main():
    input_dir = '../input/'
    output_dir = './eval_pairs.csv'
    exclude_files = ['train_narratives_colorectal.txt', 'train_narratives_gastric.txt', 'train_narratives_pancreatic.txt']
    
    process_directory(input_dir, exclude_files, output_dir)

if __name__ == "__main__":
    main()
