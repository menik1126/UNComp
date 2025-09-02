from datasets import load_dataset, Features, Value, Sequence

# Define the features schema
ft = Features({
    "id": Value("int64"),
    "context": Value("string"),
    "input": Value("string"),
    "answer": Sequence(Value("string")),
    "options": Sequence(Value("string"))
})

# Load the dataset with the specified features
dataset = load_dataset("xinrongzhang2022/InfiniteBench", features=ft)
