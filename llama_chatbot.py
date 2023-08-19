import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"

def generate_response(user_message):
    response = openai.Completion.create(
        engine="llama2",  # Use the appropriate engine name
        prompt=user_message,
        max_tokens=50  # Adjust as needed
    )
    return response.choices[0].text.strip()

st.title("Llama2 Chatbot")

user_input = st.text_input("You:", "")

if st.button("Send"):
    response = generate_response(user_input)
    st.text("Llama2: " + response)

