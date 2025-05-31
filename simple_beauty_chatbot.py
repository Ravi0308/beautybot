import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

# Initialize the Gemini model
# Ensure GOOGLE_API_KEY is set in your .env file
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)

# Set up conversation memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", # This key MUST match the one in the prompt
    k=5,                      # Remember last 5 interactions
    return_messages=True      # Important for chat models
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Seraphina, a multilingual AI Beauty Advisor. You must:
    1. ALWAYS respond in the SAME LANGUAGE that the user uses to ask their question
    2. Provide helpful advice about beauty products, skincare, makeup, hair care, and fragrances
    3. If a question is not related to beauty, politely decline to answer in the user's language
    4. Give culturally appropriate product suggestions based on the user's language/region
    5. Maintain conversation context and personality across different languages
    6. For non-English queries, ensure product names are both in local language and English in parentheses
    
    You are knowledgeable about beauty products worldwide, their ingredients, and benefits for different skin types.
    Always provide practical advice and relevant product suggestions."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Define a Pydantic model for the structured output
class BeautyAdvice(BaseModel):
    advice: str = Field(description="The main beauty advice or answer to the user's question. Must be in the same language as the user's question.")
    product_suggestions: List[str] = Field(description="A list of product types or specific product suggestions. Include both local language name and English name in parentheses where applicable.", default=[])
    tips: List[str] = Field(description="Additional tips related to the beauty advice. Must be in the same language as the user's question.", default=[])
    detected_language: str = Field(description="The detected language of the user's input for reference.", default="english")

# Create a chain to combine the prompt, model, and memory
# Use with_structured_output to guide the model to return a Pydantic object
chain = prompt | llm.with_structured_output(BeautyAdvice)

# Function to get response from the chatbot
def get_beauty_response(question: str) -> BeautyAdvice:
    # Get the current chat history from memory
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Invoke the chain with the current input and chat history
    # The response will now be a BeautyAdvice object
    response_object = chain.invoke({"input": question, "chat_history": chat_history})

    # Save the current interaction to memory (store the string representation)
    # You might want to store the structured object or a specific part of it
    memory.save_context({"input": question}, {"output": response_object.model_dump_json()})

    return response_object

# Simple chat loop for interaction
if __name__ == "__main__":
    print("\n💄 Seraphina: Hi! I'm Seraphina, your AI Beauty Advisor. I can help you in any language!")
    print("How can I help you with your beauty questions today?\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        try:
            response_object = get_beauty_response(user_input)
            
            print("\n💄 Seraphina:")
            print(f"  💭 {response_object.advice}")
            
            if response_object.product_suggestions:
                print("\n  ✨ Product Suggestions:")
                for product in response_object.product_suggestions:
                    print(f"    • {product}")
            
            if response_object.tips:
                print("\n  💡 Tips:")
                for tip in response_object.tips:
                    print(f"    • {tip}")
            
            if response_object.detected_language.lower() != "english":
                print(f"\n  🌍 [{response_object.detected_language}]")
            print()  # Add extra line break for readability
            
        except Exception as e:
            print(f"\n❌ Seraphina: Oops! I encountered an error: {e}")
            print("Please try asking again?\n")
