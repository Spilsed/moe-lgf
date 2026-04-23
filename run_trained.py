import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.model import DeconstructedJetMoE, StatefulGate

def run_inference(prompt: str, model_path: str = './jetmoe-trainer/checkpoint-40') -> str:
    model_wrapper = DeconstructedJetMoE(StatefulGate, StatefulGate)

    model_wrapper.base.load_state_dict(
        load_file(f"{model_path}/model.safetensors", device="cpu"),
        strict=False,
    )

    model = model_wrapper.base.to("cpu").eval()
    tokenizer = model_wrapper.tokenizer

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    for m in model.modules():
        if isinstance(m, StatefulGate):
            m.reset_state()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.8,
            do_sample=True
        )
    
    output = tokenizer.decode(outputs[0])
    return output

if __name__ == "__main__":
    print(run_inference("User:\nHow many electron shells are in a Potassium-40\nHelpful AI chatbot's answer:\n"))