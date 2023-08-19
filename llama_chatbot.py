import streamlit as st
import streamlit.components.v1 as components
import openai

# Adding Image to web app
st.set_page_config(page_title="llama_chatbot",layout="wide",initial_sidebar_state="auto")


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


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

