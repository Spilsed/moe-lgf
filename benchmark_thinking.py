#!/usr/bin/python
import torch
import signal
import sys
import time
import re
from datasets import load_dataset
from safetensors.torch import load_file
from src.model.model import DeconstructedJetMoE, StatefulGate

# --- Configuration ---
# Firm instructions to force the "Reasoning -> Final Answer" flow
sysPrompt = """Solve the following multiple-choice question. 
First, explain your reasoning step-by-step. 
Second, based on your reasoning, provide the final answer.
Format the end of your response exactly like this: 'The correct answer is [LABEL]' where [LABEL] is the letter or number of the choice."""

model_path = './jetmoe-trainer/checkpoint-40'
num_rows = 100
whole_set = False
split = "test"
subset = "ARC-Challenge"
dataset = "allenai/ai2_arc"

# --- CLI Handling ---
if '-c' in sys.argv:
    if (a := sys.argv[1 + sys.argv.index("-c")]) == "all": whole_set = True
    else: num_rows = int(a)
if '-s' in sys.argv: split = sys.argv[1 + sys.argv.index("-s")]
if '-ss' in sys.argv: subset = sys.argv[1 + sys.argv.index("-ss")]

# --- Model Initialization ---
print(f"Loading custom JetMoE model from {model_path}...")
model_wrapper = DeconstructedJetMoE(StatefulGate, StatefulGate)
model_wrapper.base.load_state_dict(
    load_file(f"{model_path}/model.safetensors", device="cpu"),
    strict=False,
)
model = model_wrapper.base.to("cpu").eval()
tokenizer = model_wrapper.tokenizer

def run_inference_for_grading(question: str, choices: dict) -> str:
    formatted_choices = '\n'.join(map(': '.join, zip(choices["label"], choices["text"])))
    
    # Structure the message for Chain-of-Thought
    messages = [
        {"role": "system", "content": sysPrompt},
        {"role": "user", "content": f"Question: {question}\n\nChoices:\n{formatted_choices}"}
    ]
    
    try:
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{sysPrompt}\n\nUser: {question}\n{formatted_choices}\nAssistant: Reasoning:"
    except:
        prompt = f"{sysPrompt}\n\nUser: {question}\n{formatted_choices}\nAssistant: Reasoning:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    input_length = inputs['input_ids'].shape[-1]

    # Clear memory for the MoE Gates
    for m in model.modules():
        if isinstance(m, StatefulGate):
            m.reset_state()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,    # Plenty of room for reasoning
            temperature=0.7,       # A bit of creativity helps logic flow
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

def extract_label(text: str) -> str:
    """
    Finds the answer label in a reasoning-heavy response.
    Prioritizes the 'correct answer is' pattern at the end.
    """
    # Look for the specific 'correct answer is X' pattern (case insensitive)
    match = re.search(r"correct answer is:?\s*([A-Z0-9])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: find the very last standalone character (A, B, C, D or 1, 2, 3, 4)
    # This works if the model just ends with 'The answer is B' or just 'B'
    last_char_match = re.findall(r"\b([A-Z0-9])\b", text)
    if last_char_match:
        return last_char_match[-1].upper()
    
    return ""

def send_and_grade(question: str, choices: dict, answer: str):
    print("\n" + "—"*40)
    print(f"Q: {question}")
    print('\n'.join(map(': '.join, zip(choices["label"], choices["text"]))))
    
    start_time = time.time()
    raw_response = run_inference_for_grading(question, choices)
    elapsed = time.time() - start_time
    
    predicted = extract_label(raw_response)
    
    # Print the full thought process so we can see the logic
    print(f"\n[MODEL REASONING]\n{raw_response}")
    print(f"\nExtracted: {predicted} | Actual: {answer} | Time: {elapsed:.2f}s")

    # Handle numeric/alpha mapping
    mapping = {"1":"A", "2":"B", "3":"C", "4":"D", "A":"1", "B":"2", "C":"3", "D":"4"}
    
    is_correct = (predicted == answer) or (predicted in mapping and mapping[predicted] == answer)

    if is_correct:
        print("Result: ✅ CORRECT")
        return True
    else:
        print("Result: ❌ INCORRECT")
        return False

# --- Main Data Loop ---
print(f"Fetching {subset} dataset...")
data = load_dataset(dataset, subset, split=split).shuffle(seed=int(time.time()))
if not whole_set:
    data = data.take(num_rows)

num_correct = 0
total = 0
paused = False

def handle_int(signum, frame):
    global paused
    paused = True
signal.signal(signal.SIGINT, handle_int)

for item in data:
    if paused:
        if input("\n[PAUSED] Quit? y/[n]: ").lower() == 'y': break
        paused = False
        print("Resuming...")

    total += 1
    print(f"\nProcessing {total}/{num_rows if not whole_set else 'all'}")
    
    if send_and_grade(item["question"], item["choices"], item["answerKey"]):
        num_correct += 1
    
    print(f"Current Accuracy: {num_correct/total*100:.2f}%")

print(f"\n{'='*20}\nFINAL RESULTS: {num_correct}/{total} ({num_correct/total*100:.2f}%)")