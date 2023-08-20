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

# Initialize the llama2 model
llama2_model = Llama2Model()  # Replace with the actual initialization code

# Function to perform predictions using the llama2 model
def llama2_predict(input_text):
    llama2_output = llama2_model.predict(input_text)  # Replace with the actual prediction code
    return llama2_output

# Streamlit app
def main():
    st.title("llama2 Model App")
    
    # Input text box
    input_text = st.text_area("Enter some input text:", "")
    
    if st.button("Predict"):
        if input_text:
            prediction = llama2_predict(input_text)
            st.success(prediction)
        else:
            st.warning("Please enter some input text.")

if __name__ == "__main__":
    main()
