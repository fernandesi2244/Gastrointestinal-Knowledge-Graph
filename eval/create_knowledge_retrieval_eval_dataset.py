import os
import random
import csv
from openai import OpenAI
from dotenv import load_dotenv
import argparse
from tqdm import tqdm

load_dotenv()

def get_random_chunks(file_path, num_chunks=3):
    """Extract random chunks from a text file, with each chunk being roughly 3 sentences."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Simple sentence splitting using periods, question marks, and exclamation points
        # This is a simplified approach compared to NLTK
        sentences = []
        for raw_sentence in text.split('.'):
            # Further split by ! and ?
            for s in raw_sentence.split('!'):
                for s2 in s.split('?'):
                    if s2.strip():
                        sentences.append(s2.strip() + '.')
        
        # Remove empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        # If not enough sentences, return the whole text
        if len(sentences) <= 3:
            return [text]
        
        chunks = []
        for _ in range(num_chunks):
            # Select a random starting point, leaving room for 3 sentences
            if len(sentences) <= 3:
                start_idx = 0
            else:
                start_idx = random.randint(0, len(sentences) - 3)
            
            # Create a chunk of approximately 3 sentences
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
