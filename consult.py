import subprocess
from typing import List
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import os

MODEL_ID = "gpt-4.1-nano"
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

For any questions requiring knowledge retrieval, primarily use the output from the medical database.
This is important, as a complicated procedure has been set up to query the relevant information from the
database for you. The database is a comprehensive resource for gastrointestinal cancers.

Remember to never make up information. If you're unsure, acknowledge your limitations and suggest 
consulting with a healthcare professional."""

@dataclass
class Message:
    role: str
    content: str

class GICancerChatbot:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('GRAPHRAG_API_KEY')
        if api_key is None:
            raise ValueError("GRAPHRAG_API_KEY not found in environment variables.")
        
        self.client = OpenAI(api_key=api_key)
        self.chat_history: List[Message] = []
        print("OpenAI client initialized successfully. Starting chat session...")
        
    def format_chat_history(self):
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
        
        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=decision_prompt,
            temperature=0.0,
            max_tokens=10
        )
        
        decision = response.choices[0].message.content.strip()
        
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
        print('Determined search type:', search_type)
        
        graphrag_response = self.query_graphrag(user_input, search_type)
        print('Got response from GraphRAG')
        
        self.chat_history.append(Message(role="system", content=f"Here's information from the medical database: {graphrag_response}"))
        
        formatted_messages = self.format_chat_history()
        
        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=formatted_messages,
            temperature=TEMPERATURE
        )
        
        assistant_response = response.choices[0].message.content
        print('Got response from OpenAI')
        
        self.chat_history.append(Message(role="assistant", content=assistant_response))
        
        return assistant_response
    
    def run_cli(self):
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        while True:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ["quit", "exit", "q", "stop"]:
                print("\nThank you for using the GI Cancer Chatbot. Take care!")
                break
                
            print("\nProcessing your question...")
            response = self.generate_response(user_input)
            print(f"\n🤖 Assistant: {response}")

def main():
    load_dotenv()
    chatbot = GICancerChatbot()
    chatbot.run_cli()

if __name__ == "__main__":
    main()