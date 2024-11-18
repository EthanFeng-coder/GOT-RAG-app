from unsloth import FastLanguageModel
import torch

# Model hyperparameters
max_seq_length = 2048
dtype = None  # Auto-detect precision (Bfloat16, etc.)
load_in_4bit = True  # Use 4-bit quantization for memory efficiency

# Load fine-tuned LoRA model and tokenizer from the local path
local_model_path = "lora_model"  # Replace with your local LoRA model directory

# Load model and tokenizer from the local path
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=local_model_path,  # Load from local path
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Switch model to inference mode (for generating output)
FastLanguageModel.for_inference(model)

# Prepare the prompt input directly as a string
prompt = "In the book a song of ice and fire who is the sword of the morning?"

# Tokenize the prompt
inputs = tokenizer(
    prompt,
    return_tensors="pt",  # Return tensor format for PyTorch
    padding=True,
    truncation=True,
    max_length=max_seq_length  # Ensure max sequence length is respected
).to("cuda")  # Move inputs to GPU

# Generate output (adjust max_new_tokens as needed)
outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=64, use_cache=True)

# Decode the generated output to human-readable text
output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Print the generated text
print("Generated output:", output_texts)
