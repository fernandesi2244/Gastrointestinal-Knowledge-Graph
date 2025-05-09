import pandas as pd
import os
import sys
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import json
import re

# consult is actually ../consult.py, so need to add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consult import GICancerChatbot

EVALUATION_MODEL = "gpt-4.1-nano"
EVALUATION_TEMPERATURE = 0.0  # Use 0 for consistent extraction

EXTRACTION_PROMPT = """Given the following chatbot response to a cancer type prediction request, extract the cancer type prediction.

Chatbot response: {chatbot_response}

The chatbot should have predicted one of these cancer types: colorectal, gastric, pancreatic.

Extract and return ONLY the predicted cancer type from the response. Return exactly one of these words:
- colorectal
- gastric  
- pancreatic

If no clear prediction is made or multiple predictions are given, return "none".
"""

class CancerTypePredictionEvaluator:
    def __init__(self, csv_path: str):
        load_dotenv()
        api_key = os.getenv('GRAPHRAG_API_KEY')
        
        self.client = OpenAI(api_key=api_key)
        self.csv_path = csv_path
        self.results = []
        self.cancer_types = ['colorectal', 'gastric', 'pancreatic']
            
    def create_prediction_prompt(self, narrative: str) -> str:
        """Create a prompt that asks the chatbot to predict cancer type"""
        prompt = f"""Based on the following patient narrative, please predict which type of gastrointestinal cancer this patient most likely has. Choose from: colorectal, gastric, or pancreatic cancer.

Patient narrative:
{narrative}

Please provide your prediction along with your reasoning. Which of the three cancer types (colorectal, gastric, or pancreatic) is most likely based on the symptoms and information provided?"""
        return prompt
    
    def get_chatbot_prediction(self, narrative: str) -> str:
        try:
            chatbot = GICancerChatbot()
            prompt = self.create_prediction_prompt(narrative)
            response = chatbot.generate_response(prompt)
            return response
        except Exception as e:
            print(f"Error getting chatbot response: {e}")
            return f"Error: {str(e)}"
    
    def extract_prediction(self, chatbot_response: str) -> str:
        extraction_prompt = EXTRACTION_PROMPT.format(chatbot_response=chatbot_response)
        
        response = self.client.chat.completions.create(
            model=EVALUATION_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at extracting cancer type predictions from text. Return only the cancer type."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=EVALUATION_TEMPERATURE,
            max_tokens=10
        )
        
        prediction = response.choices[0].message.content.strip().lower()
        
        # Validate prediction
        if prediction in self.cancer_types:
            return prediction
        
        return "none"
    
    def run_evaluation(self) -> Dict:
        df = pd.read_csv(self.csv_path)
        total_cases = len(df)
        
        print(f"Starting evaluation of {total_cases} patient narratives...")
        
        correct_predictions = 0
        predictions_by_type = {ct: {'correct': 0, 'total': 0} for ct in self.cancer_types}
        confusion_matrix = {actual: {predicted: 0 for predicted in self.cancer_types + ['none']} 
                          for actual in self.cancer_types}
        
        for index, row in df.iterrows():
            narrative = row['Narrative']
            actual_type = row['Cancer Type']
            
            print(f"\nCase {index + 1}/{total_cases}")
            print(f"Actual cancer type: {actual_type}")

            print("Getting chatbot prediction...")
            chatbot_response = self.get_chatbot_prediction(narrative)

            predicted_type = self.extract_prediction(chatbot_response)
            print(f"Predicted cancer type: {predicted_type}")

            is_correct = (predicted_type == actual_type)
            if is_correct:
                correct_predictions += 1
            
            if actual_type in self.cancer_types:
                predictions_by_type[actual_type]['total'] += 1
                if is_correct:
                    predictions_by_type[actual_type]['correct'] += 1
                
                confusion_matrix[actual_type][predicted_type] += 1
            
            result = {
                'narrative': narrative,
                'actual_type': actual_type,
                'predicted_type': predicted_type,
                'chatbot_response': chatbot_response,
                'is_correct': is_correct
            }
            self.results.append(result)
        
        overall_accuracy = correct_predictions / total_cases if total_cases > 0 else 0
        
        # Calculate per-type accuracy
        type_accuracies = {}
        for cancer_type in self.cancer_types:
            total = predictions_by_type[cancer_type]['total']
            correct = predictions_by_type[cancer_type]['correct']
            type_accuracies[cancer_type] = correct / total if total > 0 else 0
        
        summary = {
            'total_cases': total_cases,
            'correct_predictions': correct_predictions,
            'overall_accuracy': overall_accuracy,
            'accuracy_by_type': type_accuracies,
            'predictions_by_type': predictions_by_type,
            'confusion_matrix': confusion_matrix
        }
        
        return summary
    
    def save_results(self, output_path: str = 'cancer_prediction_results.json'):
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to {output_path}")
    
    def print_confusion_matrix(self, confusion_matrix: Dict):
        print("\nConfusion Matrix:")
        print("Actual\\Predicted", end="\t")
        for pred_type in self.cancer_types + ['none']:
            print(pred_type, end="\t")
        print()
        
        for actual_type in self.cancer_types:
            print(actual_type, end="\t\t")
            for pred_type in self.cancer_types + ['none']:
                count = confusion_matrix[actual_type][pred_type]
                print(count, end="\t")
            print()

def main():
    csv_path = 'numeric_data/test_narratives.csv'
    
    evaluator = CancerTypePredictionEvaluator(csv_path)
    summary = evaluator.run_evaluation()
    
    # Print summary
    print("\n" + "="*50)
    print("CANCER TYPE PREDICTION EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Cases: {summary['total_cases']}")
    print(f"Correct Predictions: {summary['correct_predictions']}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
    
    print("\nAccuracy by Cancer Type:")
    for cancer_type, accuracy in summary['accuracy_by_type'].items():
        total = summary['predictions_by_type'][cancer_type]['total']
        correct = summary['predictions_by_type'][cancer_type]['correct']
        print(f"  {cancer_type.capitalize()}: {accuracy:.2%} ({correct}/{total})")
    
    # Print confusion matrix
    evaluator.print_confusion_matrix(summary['confusion_matrix'])
    
    # Save detailed results
    evaluator.save_results()

if __name__ == "__main__":
    main()