# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "D:\emotional_chatbot\chatbot_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=False 
    )

    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)



# Streamlit UI

st.title("🤖 Chatbot")

# Store conversation in session
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:", "")

if st.button("Send") and user_input.strip() != "":
    # Append user input to history
    st.session_state.history.append(f"A: {user_input}")

    # Build context string for chatbot
    context = "\n".join(st.session_state.history) + "\nB:"

    # Tokenize and move to same device as model
    inputs = tokenizer(context, return_tensors="pt").to(device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[1] + 100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("B:")[-1].strip()

    # Append bot response to history
    st.session_state.history.append(f"B: {response}")

# Display conversation
for msg in st.session_state.history:
    if msg.startswith("A:"):
        st.markdown(f"**You:** {msg[3:]}")
    else:
        st.markdown(f"**Bot:** {msg[3:]}")