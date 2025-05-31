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
    ("system", "You are Seraphina, a friendly AI Beauty Advisor. Your sole purpose is to provide helpful advice and information ONLY about beauty products, skincare, makeup, hair care, and fragrances. If a question is not related to beauty, politely decline to answer and remind the user of your purpose."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Define a Pydantic model for the structured output
class BeautyAdvice(BaseModel):
    advice: str = Field(description="The main beauty advice or answer to the user's question.")
    product_suggestions: List[str] = Field(description="A list of product types or specific product suggestions related to the advice.", default=[])
    tips: List[str] = Field(description="Additional tips related to the beauty advice.", default=[])

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
    print("Seraphina: Hi! I'm Seraphina, your AI Beauty Advisor. How can I help you with your beauty questions today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        try:
            response_object = get_beauty_response(user_input)
            print("\U0001F484 Seraphina:")
            print(f"  Advice: {response_object.advice}")
            if response_object.product_suggestions:
                print("  Product Suggestions:")
                for product in response_object.product_suggestions:
                    print(f"    - {product}")
            if response_object.tips:
                print("  Tips:")
                for tip in response_object.tips:
                    print(f"    - {tip}")
        except Exception as e:
            print(f"Seraphina: Oops! I encountered an error: {e}")
            print("Seraphina: Could you please try asking again?")
