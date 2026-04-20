from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=100)
print(result)