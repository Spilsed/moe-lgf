import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "./inc/jetmoe-local"

# 1. Manually load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)

# 2. Manually load the model with optimized settings
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 3. Create the pipeline using the objects we just made
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer
)

try:
    while p:=input('\n>'):
        content = [
            {"role": "user", "content": p},
        ]
        pipe(content, max_new_tokens=512, do_sample=True, temperature=1)
except EOFError:
    print("quit\nGoodbye!")