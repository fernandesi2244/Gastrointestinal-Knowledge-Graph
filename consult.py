import subprocess
from typing import List
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

LLAMA4_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 2048
TEMPERATURE = 0.2 # TODO: prob experiment with this a bit

SYSTEM_PROMPT = """You are a highly skilled medical assistant specializing in gastrointestinal cancers.
Your role is to be exceptionally helpful, caring, and attentive to patients seeking information about 
gastrointestinal cancers, including causes, symptoms, treatments, and other related aspects.

You should respond in the manner of an exceptional doctor: knowledgeable, compassionate, and 
patient-focused. Ask clarifying questions when needed to ensure you're providing the most relevant 
information for the patient's specific concerns.

When responding:
1. Be empathetic and acknowledge the patient's concerns
2. Provide accurate, evidence-based information using the medical resources available to you
3. Explain complex medical concepts in clear, accessible language
4. Suggest relevant questions the patient might not have thought to ask
5. Always clarify that you are an AI assistant and recommend consulting with healthcare professionals

For any questions requiring knowledge retrieval, you'll use GraphRAG:
- For general/aggregate information about gastrointestinal cancers, use the GLOBAL search option
- For specific details about particular cancers/conditions, use the LOCAL search option

Remember to never make up information. If you're unsure, acknowledge your limitations and suggest 
consulting with a healthcare professional."""

@dataclass
class Message:
    role: str
    content: str

class GICancerChatbot:
    def __init__(self):        
        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA4_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLAMA4_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None
        )
        
        self.chat_history: List[Message] = []
        print("Model loaded successfully. Starting chat session...")
        
    def format_chat_history(self) -> str:
        formatted_messages = []
        
        formatted_messages.append({"role": "system", "content": SYSTEM_PROMPT})

        for message in self.chat_history:
            formatted_messages.append({"role": message.role, "content": message.content})
            
        return formatted_messages
    
    def determine_search_type(self, query: str) -> str:
        decision_prompt = [
            {"role": "system", "content": "You are a helpful assistant that determines the appropriate search type for medical queries. For general or aggregate-level questions about gastrointestinal cancers as a category, recommend 'GLOBAL' search. For specific questions about particular conditions, symptoms, treatments, or details, recommend 'LOCAL' search. Reply with ONLY the word 'GLOBAL' or 'LOCAL'."},
            {"role": "user", "content": f"Determine the appropriate search type for this query about gastrointestinal cancer: '{query}'"}
        ]
        
        inputs = self.tokenizer.apply_chat_template(decision_prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(inputs, max_length=MAX_LENGTH, temperature=0.1, top_p=0.9)
        decision = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "GLOBAL" in decision.upper():
            return "global"
        else:
            return "local" # pretty good default since user probably has a specific question in mind
    
    def query_graphrag(self, query: str, search_type: str) -> str:
        command = f"graphrag query --root . --method {search_type} --query \"{query}\""
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Error querying GraphRAG: {result.stderr.strip()}")
            return "I'm sorry, but I couldn't retrieve the information at this time."
    
    def generate_response(self, user_input: str) -> str:
        self.chat_history.append(Message(role="user", content=user_input))
        
        search_type = self.determine_search_type(user_input)
        
        graphrag_response = self.query_graphrag(user_input, search_type)
        
        self.chat_history.append(Message(role="system", content=f"Here's information from the medical database: {graphrag_response}"))
        
        formatted_messages = self.format_chat_history()
        
        inputs = self.tokenizer.apply_chat_template(formatted_messages, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(
            inputs, 
            max_length=MAX_LENGTH, 
            temperature=TEMPERATURE,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response (everything after the prompt)
        # TODO: see if this is robust enough
        print('Response from LLAMA:', response)
        response = response.split("Here's information from the medical database:")[-1].split("Assistant:")[-1].strip()
        
        self.chat_history.append(Message(role="assistant", content=response))
        
        return response
    
    def run_cli(self):
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        while True:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nThank you for using the GI Cancer Chatbot. Take care!")
                break
                
            print("\nProcessing your question...")
            response = self.generate_response(user_input)
            print(f"\n🤖 Assistant: {response}")

def main():    
    chatbot = GICancerChatbot()
    chatbot.run_cli()

if __name__ == "__main__":
    main()