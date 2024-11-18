import os
import torch
import time  # Import time for sleep functionality
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template
from unsloth import is_bfloat16_supported

# Hyperparameters
max_seq_length = 2048  # Maximum sequence length for the model
dtype = None  # Automatically detected, use Float16 for Tesla T4, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4-bit quantization to reduce memory usage

# Load pre-trained LLaMA 3.2-3B model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

# Track GPU memory usage
start_gpu_memory = torch.cuda.memory_reserved() / 1024 / 1024 / 1024  # In GB
max_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # Total GPU memory

# Apply LoRA for fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # No dropout
    bias="none",  # No bias
    use_gradient_checkpointing="unsloth",  # Enable gradient checkpointing for memory efficiency
    random_state=3407
)

# Get the appropriate chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Connect to Milvus
connections.connect(
    alias="default",  # Alias for the connection, can be "default"
    host="127.0.0.1",  # Make sure this is the correct host IP of your Milvus instance
    port="19530"  # This is the default Milvus port, ensure it's correct
)

# Define the collection name
collection_name = "song_of_ice_and_fire"

# Check if the collection exists, and handle it appropriately
if utility.has_collection(collection_name):
    collection = Collection(name=collection_name)  # Initialize the collection
    print(f"Connected to existing collection '{collection_name}'.")
else:
    # Define schema and create the collection if it does not exist
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Auto-generated IDs
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # Embedding size of 1024 dimensions
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000)  # Field to store the text
    ]
    schema = CollectionSchema(fields, description="Embeddings and Text of Song of Ice and Fire PDFs",
                              enable_dynamic_field=True)

    # Create the collection
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection '{collection_name}'.")

limit = 10000  # Number of records to fetch per batch (you can adjust this)
offset = 0  # Start at the first record
all_texts = []  # List to store all the results

max_offset = 54293  # Max allowed value for offset + limit

# Fetch all records from Milvus in batches
while offset + limit <= max_offset:
    try:
        results = collection.query(expr="", output_fields=["text"], offset=offset, limit=limit)  # Fetch batch
        if not results:  # Break if no more records are returned
            break
        texts = [result["text"] for result in results]
        all_texts.extend(texts)  # Append to the final list
        offset += limit  # Increment offset to fetch the next batch
        time.sleep(1)  # Sleep for 1 second after each batch fetch
    except Exception as e:
        print(f"Error during query: {e}")
        break

print(f"Fetched a total of {len(all_texts)} records.")

# Create the dataset with the fetched texts
dataset = {"text": all_texts}  # Ensure correct key 'text'
dataset = Dataset.from_dict(dataset)  # Create Hugging Face dataset

# Define function to format the prompts and add labels for the training
def formatting_prompts_func(examples):
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_seq_length)
    return {"input_ids": inputs["input_ids"], "labels": inputs["input_ids"]}  # Labels are the same as input_ids

# Apply formatting function
dataset = dataset.map(formatting_prompts_func)

# Prepare the data collator for padding and truncation
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

# Set up the SFTTrainer for fine-tuning
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="input_ids",  # Updated to match the formatted prompts
    max_seq_length=max_seq_length,
    data_collator=data_collator,
    dataset_num_proc=2,  # Number of processes for loading data
    packing=False,  # Disable packing
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        warmup_steps=500,
        max_steps=20000,  # Total steps for training
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",  # Use 8-bit Adam optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_steps=100,  # Save a checkpoint every 100 steps
        save_total_limit=5,  # Keep only the last 5 checkpoints
        report_to="none"
    )
)

# Perform fine-tuning
for step in range(0, 20000, 100):
    trainer_stats = trainer.train()  # Run the training
    print(f"Completed step {step}, sleeping for 5 seconds...")
    time.sleep(60)  # Sleep for 5 seconds after every 100 steps

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)  # In GB
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save the fine-tuned model and tokenizer
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

print("Fine-tuning complete. Model and tokenizer saved to 'lora_model'.")
