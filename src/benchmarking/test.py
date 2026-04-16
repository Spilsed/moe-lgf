import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator

torch.set_num_threads(12)

accelerator=Accelerator()
model_id = "./inc/jetmoe-local"

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    local_files_only=True,
    trust_remote_code=True,
    device_map={"":accelerator.device}
)

try:
    while p:=input('\n>'):
        content = [
            {"role": "user", "content": p},
        ]
        prompt = tokenizer.apply_chat_template(
            content,
            tokenize=False,
            add_generation_prompt=True
        )
        m_input=tokenizer(prompt,return_tensors='pt').to(accelerator.device)
        gen_ids=model.generate(
            **m_input,
            do_sample=True,
            temperature=.8,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
        input_length=m_input['input_ids'].shape[-1]
        response=tokenizer.decode(gen_ids[0][input_length:],skip_special_tokens=True).strip()
        print(response)
except EOFError:
    print("quit\nGoodbye!")