import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "jetmoe/jetmoe-8b", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b")

model: transformers.JetMoeModel = base.model

inputs = tokenizer("Hello", return_tensors="pt")

outputs = base.generate(
    **inputs,
    max_new_tokens=1,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print(model.layers._modules)
