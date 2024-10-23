import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the alpaca prompt template
alpaca_prompt = "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"

# Load the model
@st.cache_resource
def load_model():
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/meta-llama-3.1-8b-bnb-4bit"  # Base model
    )

    # Apply dynamic quantization
    model = torch.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/meta-llama-3.1-8b-bnb-4bit")

    return model, tokenizer

# Streamlit App Interface
st.title("My Little Broker LLM")

# User input for the prompt
user_prompt = st.text_area("Enter your prompt:", value="Where can I find the online listing for the building in Greater Cairo / Bait El Watan El Asasy?")

# Load the model and tokenizer
model, tokenizer = load_model()

if st.button("Generate Response"):
    # Define your prompt using the alpaca format
    prompt = alpaca_prompt.format(user_prompt, "", "")

    # Tokenize the input prompt
    inputs = tokenizer([prompt], return_tensors="pt")

    # Generate the output (text generation)
    with st.spinner("Generating response..."):
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

    # Decode and clean the generated text
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    generated_text = generated_text.split("### Response:")[1].strip()

    # Display the result
    st.subheader("Generated Response:")
    st.write(generated_text)
