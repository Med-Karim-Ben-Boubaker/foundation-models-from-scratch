from src.data.datasets import GPTDatasetV1
from src.data.tokenizer import get_tokenizer

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    
    # Test case 1: Short text with max_length too large
    print("=== Test Case 1: Short text with max_length too large ===")
    text1 = "Hello, world!"
    dataset1 = GPTDatasetV1(text=text1, tokenizer=tokenizer, max_length=10, stride=1)
    print(f"Dataset length: {len(dataset1)}")
    
    # Test case 2: Short text with appropriate max_length
    print("\n=== Test Case 2: Short text with appropriate max_length ===")
    dataset2 = GPTDatasetV1(text=text1, tokenizer=tokenizer, max_length=3, stride=1)
    print(f"Dataset length: {len(dataset2)}")
    if len(dataset2) > 0:
        input_seq, target_seq = dataset2[0]
        print(f"First input sequence: {input_seq}")
        print(f"First target sequence: {target_seq}")
    
    # Test case 3: Longer text with multiple sequences
    print("\n=== Test Case 3: Longer text with multiple sequences ===")
    text3 = "Hello, world! This is a longer text for testing."
    dataset3 = GPTDatasetV1(text=text3, tokenizer=tokenizer, max_length=3, stride=1)
    print(f"Dataset length: {len(dataset3)}")
    if len(dataset3) > 0:
        print("All sequences:")
        for i in range(len(dataset3)):
            input_seq, target_seq = dataset3[i]
            print(f"  {i}: input={input_seq.tolist()}, target={target_seq.tolist()}")