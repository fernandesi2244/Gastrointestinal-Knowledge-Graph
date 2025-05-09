import pandas as pd
import os
import sys
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import json

# NOTE: RUN THIS FILE FROM ROOT

# consult is actually ../consult.py, so need to add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consult import GICancerChatbot

EVALUATION_MODEL = "gpt-4.1-nano"
EVALUATION_TEMPERATURE = 0.0  # Use 0 for consistent evaluation

EVALUATION_PROMPT = """You are evaluating a medical chatbot's response for knowledge retention.
Given the following:
1. Question: {question}
2. Expected knowledge (text chunk): {text_chunk}
3. Chatbot's response: {chatbot_response}

Rate how accurately the chatbot's response reflects the knowledge in the text chunk.
Focus ONLY on knowledge retention and accuracy, not on style, tone, or manner of delivery.

Use this scoring scale:
5 - Excellent: The response accurately captures all key information from the text chunk
4 - Good: The response captures most key information with minor omissions
3 - Fair: The response captures some key information but misses important details
2 - Poor: The response captures minimal key information or has significant errors
1 - Very Poor: The response fails to capture the relevant information or is largely incorrect

Provide your score as a single integer between 1 and 5.
"""

class ChatbotEvaluator:
    def __init__(self, csv_path: str):
        load_dotenv()
        api_key = os.getenv('GRAPHRAG_API_KEY')
        
        self.client = OpenAI(api_key=api_key)
        self.csv_path = csv_path
        self.results = []
        
    def get_chatbot_response(self, question: str) -> str:
        chatbot = GICancerChatbot()
        response = chatbot.generate_response(question)
        return response
    
    def evaluate_response(self, question: str, text_chunk: str, chatbot_response: str) -> int:
        try:
            evaluation_prompt = EVALUATION_PROMPT.format(
                question=question,
                text_chunk=text_chunk,
                chatbot_response=chatbot_response
            )
            
            response = self.client.chat.completions.create(
                model=EVALUATION_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for medical chatbot responses. Provide only a score between 1 and 5."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=EVALUATION_TEMPERATURE,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Extract the numeric score
            try:
                score = int(score_text)
                if score < 1:
                    print(f"Score {score} is less than 1. Defaulting to 1.")
                    return 1
                elif score > 5:
                    print(f"Score {score} is greater than 5. Defaulting to 5.")
                    return 5
                return score
            except ValueError:
                print(f"Invalid score format: {score_text}. Defaulting to 3.")
                return 3
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 3
    
    def run_evaluation(self) -> Dict:
        df = pd.read_csv(self.csv_path)
        total_questions = len(df)
        
        print(f"Starting evaluation of {total_questions} questions...")
        
        for index, row in df.iterrows():
            question = row['question']
            text_chunk = row['text_chunk']
            
            print(f"\nQuestion {index + 1}/{total_questions}: {question[:50]}...")
            
            print("Getting chatbot response...")
            chatbot_response = self.get_chatbot_response(question)
            
            print("Evaluating response...")
            score = self.evaluate_response(question, text_chunk, chatbot_response)
            
            result = {
                'question': question,
                'text_chunk': text_chunk,
                'chatbot_response': chatbot_response,
                'score': score
            }
            self.results.append(result)
        
        # Calculate average score
        scores = [r['score'] for r in self.results]
        average_score = sum(scores) / len(scores) if scores else 0
        
        summary = {
            'total_questions': total_questions,
            'average_score': average_score,
            'score_distribution': {
                '1': scores.count(1),
                '2': scores.count(2),
                '3': scores.count(3),
                '4': scores.count(4),
                '5': scores.count(5)
            }
        }
        
        return summary
    
    def save_results(self, output_path: str = 'eval/evaluation_results.json'):
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to {output_path}")

def main():
    csv_path = 'eval/eval_pairs.csv'
    
    evaluator = ChatbotEvaluator(csv_path)
    summary = evaluator.run_evaluation()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Average Score: {summary['average_score']:.2f}/5")
    print("\nScore Distribution:")
    for score, count in summary['score_distribution'].items():
        print(f"  Score {score}: {count} questions")
    
    # Save detailed results
    evaluator.save_results()

if __name__ == "__main__":
    main()