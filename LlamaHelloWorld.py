from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your Hugging Face token
hf_token = "hftoken"

model_name = "meta-llama/Llama-2-7b-hf"

# Load the model and tokenizer with authentication token
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",  # Use CPU instead of GPU
    token=hf_token
)

# Define your input
input_text = "Analyze the sentiment of this review: 'The product is amazing!'"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate a response
outputs = model.generate(inputs["input_ids"], max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
