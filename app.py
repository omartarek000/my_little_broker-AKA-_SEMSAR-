import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# Define the alpaca prompt template
alpaca_prompt = "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"

# Load the base model
@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/meta-llama-3.1-8b-bnb-4bit",  # Base model
        load_in_4bit=False,                     # Disable 4-bit quantization for CPU
        device_map="auto"                      # This will still load the model onto CPU
    )

    # Load the PEFT fine-tuned model
    model = PeftModel.from_pretrained(base_model, "tarek009/my_little_broker")  # Your fine-tuned PEFT model

    # Optional: Convert the model to half-precision for faster inference (fp16), but ensure it's supported on CPU
    model.float()  # Use float precision instead of half

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/meta-llama-3.1-8b-bnb-4bit")

    return model, tokenizer

# Streamlit App Interface
st.title("My Little Broker LLM")

# User input for the prompt
user_prompt = st.text_area("Enter your prompt:", value="Where can I find the online listing for the building in Greater Cairo  /  Bait El Watan El Asasy?")

# Load the model and tokenizer
model, tokenizer = load_model()

if st.button("Generate Response"):
    # Define your prompt using the alpaca format
    prompt = alpaca_prompt.format(user_prompt, "", "")

    # Tokenize the input prompt and keep it on CPU
    inputs = tokenizer([prompt], return_tensors="pt")  # No need to move to GPU

    # Generate the output (text generation)
    with st.spinner("Generating response..."):
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

    # Decode and clean the generated text
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    generated_text = generated_text.split("### Response:")[1].strip()

    # Display the result
    st.subheader("Generated Response:")
    st.write(generated_text)
