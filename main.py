import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "jetmoe/jetmoe-8b", dtype="auto", device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", device_map="cpu")

model: transformers.JetMoeModel = base.model

inputs = tokenizer("User:\nWhat is the color of an apple\nHelpful AI chatbot's answer:\n", return_tensors="pt")
print(inputs)

outputs = base.generate(
    **inputs,
    max_new_tokens=32,
    do_sample=True,
    temperature=0.1,
    top_p=0.9
)

print(base)

print(outputs)
print(tokenizer.decode(outputs))
# print(len(model.layers))
# print(model.layers._modules)
