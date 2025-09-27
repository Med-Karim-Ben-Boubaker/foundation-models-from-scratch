from datasets import load_dataset

print("Starting to load dataset")
# Correct way: Specify split and streaming=True
dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

# Process more samples
for i, sample in enumerate(dataset):
    if i >= 10:  # Process first 10 samples
        break
    print(f"Sample {i}: {sample['text'][:100]}...")  # First 100 chars