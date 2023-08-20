import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import torch


st.set_page_config(page_title="llama_chatbot_d",layout="wide",initial_sidebar_state="auto")


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


output_merged_dir = 'final_merged_checkpoint'
device_map = {"": 0}


# Load pre-trained model checkpoint and tokenizer


# Get the tokenizer
tokenizer = AutoTokenizer.from_pretrained(output_merged_dir)
# cfg = get_cfg()
# cfg.MODEL.DEVICE = 'cpu'
# Load the model
model = AutoModelForCausalLM.from_pretrained(output_merged_dir, load_in_4bit=True,
 torch_dtype=torch.bfloat16,
 device_map='cuda')
model.config.use_cache = True 



prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
INSTRUCTION_KEY:
'###<instruction_start>
Extract following entities which has meaning as separated by colon 1.MORTGAGE_LAND_SHARE : mortgage land share number of property 2.LAND_SHARE :  part of land piece of an owner 3.OWNER_NAME :  name of person 4.RELATION : keywords signifying relations 5.RELATION_NAME :  name of related person 6.MORTGAGE_KHASRA_NO : khasara number of land 7.NON_MORTGAGE_KHASRA_NO : not the mortgage khasra number 8.BANK_NAME: name of bank or society 9.GUARDIAN_FLAG: flag for owner gaurdian 10.MORTGAGOR_FLAG : flag for mortgagor owner 11.CASTE:caste of owner
 <instruction_end>'
INPUT_KEY:
'###<input_start>
\nकेशर पत्नि शिवकरण हिस्सा-1/4 जाति-भाम्बी सा. देह खातेदार\n
<input_end>
"""   


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
inputs = tokenizer(prompt, return_tensors="pt", truncation=True) #.to(device) 

outputs = model.generate(input_ids=inputs["input_ids"].to(device),
                         max_new_tokens=500,
                         pad_token_id=tokenizer.eos_token_id,
                         temperature = 0.9,
                         top_k=10,
                         do_sample=True,
                         num_return_sequences=1)
                         
                         
print(f"Prompt:\n{prompt}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")    




# Set up the chatbot interface
st.title("Llama Chatbot")

user_input = st.text_input("You:", value="", max_chars=100, key="input_text")

if st.button("Chat"):
    if user_input:
        input_ids = tokenizer.encode(user_input, return_tensors="pt")

        # Generate response
        with torch.no_grad():
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.text_area("Llama:", value=response, height=100, key="response_text")

