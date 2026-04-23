import torch
import signal
import sys
import time
from datasets import load_dataset
from safetensors.torch import load_file
from src.model.model import DeconstructedJetMoE, StatefulGate

# --- Configuration ---
sysPrompt = """Choose the answer which is most correct.
Respond only with the label of the choice in the final answer.
Do not respond with any other text such as 'the correct answer is' or the text of the answer, ONLY the label(A, B, 3, etc.)
"""

model_path = './jetmoe-trainer/checkpoint-40'
num_rows = 100
whole_set = False
split = "test"
subset = "ARC-Challenge"
dataset = "allenai/ai2_arc"

# --- Handle CLI Args ---
if '-c' in sys.argv:
    if (a := sys.argv[1 + sys.argv.index("-c")]) == "all":
        whole_set = True
    else:
        num_rows = int(a)
if '-s' in sys.argv:
    split = sys.argv[1 + sys.argv.index("-s")]
if '-ss' in sys.argv:
    subset = sys.argv[1 + sys.argv.index("-ss")]

# --- Model Initialization ---
print(f"Loading custom model from {model_path}...")
# Note: Ensure DeconstructedJetMoE and StatefulGate are imported/defined
model_wrapper = DeconstructedJetMoE(StatefulGate, StatefulGate)
model_wrapper.base.load_state_dict(
    load_file(f"{model_path}/model.safetensors", device="cpu"),
    strict=False,
)
model = model_wrapper.base.to("cpu").eval()
tokenizer = model_wrapper.tokenizer

def run_inference_for_grading(question: str, choices: dict) -> str:
    # 1. Reconstruct the choice string
    formatted_choices = '\n'.join(map(': '.join, zip(choices["label"], choices["text"])))
    
    # 2. Use the Chat Template (CRITICAL for most Instruct/Chat models)
    # If your model wasn't trained with a template, change this to a simple string
    messages = [
        {"role": "system", "content": sysPrompt},
        {"role": "user", "content": f"{question}\n\nChoices:\n{formatted_choices}"}
    ]
    
    try:
        # Check if the tokenizer has a chat template, otherwise fallback to raw text
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{sysPrompt}\n\nUser: {question}\n{formatted_choices}\nAssistant:"
    except:
        prompt = f"{sysPrompt}\n\nUser: {question}\n{formatted_choices}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    input_length = inputs['input_ids'].shape[-1]

    # 3. Reset the Stateful Gates
    for m in model.modules():
        if isinstance(m, StatefulGate):
            m.reset_state()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,    # We only need 1-5 tokens for a label
            temperature=0.1,      # Lowered: early checkpoints need more "focus"
            do_sample=False,      # Greedy search is better for benchmarking accuracy
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 4. Decode ONLY the new tokens
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Slice the output to get only the part AFTER the prompt
    # If the model is still being stubborn, we look at the very last character generated
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Debug: If response is empty, the model might have failed to follow the template
    if not response:
        # Fallback: take the last token of the whole sequence if empty
        response = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True).strip()
        
    return response

def send_and_grade(question: str, choices: dict, answer: str):
    formatted_choices = '\n'.join(map(': '.join, zip(choices["label"], choices["text"])))
    full_prompt = f"{sysPrompt}\nQuestion: {question}\n{formatted_choices}\nAnswer:"
    
    print("\n" + "-"*30)
    print(f"Question: {question}")
    print(formatted_choices)
    
    start_time = time.time()
    response = run_inference_for_grading(full_prompt, choices)
    elapsed = time.time() - start_time
    
    print(f"Model response: '{response}' ({elapsed:.2f}s)")

    # Normalize logic (handling '1' vs 'A' etc)
    answer_lookup = {"1":"A", "2":"B", "3":"C", "4":"D", "A":"1", "B":"2", "C":"3", "D":"4"}
    
    if not response:
        print("Result: Empty response (Incorrect)")
        return False

    first_char = response[0].upper()
    if first_char == answer:
        print("Result: Correct")
        return True
    elif first_char in answer_lookup and answer_lookup[first_char] == answer:
        print("Result: Correct (with label replacement)")
        return True
    else:
        print(f"Result: Incorrect (Correct answer was {answer})")
        return False

# --- Data Loading ---
print(f"Loading dataset {dataset}...")
data = load_dataset(dataset, subset, split=split).remove_columns("id").shuffle(seed=int(time.time()))
if not whole_set:
    data = data.take(num_rows)

# --- Main Loop & Signal Handling ---
paused = False
def handle_int(signum, frame):
    global paused
    paused = True
signal.signal(signal.SIGINT, handle_int)

num_correct = 0
total = 0

for item in data:
    if paused:
        if input("\nPaused. Quit now? y/[n]: ").strip().lower() == 'y':
            break
        paused = False
        print("Resuming...")

    total += 1
    print(f"\nProcessing {total}/{num_rows if not whole_set else 'all'}")
    
    if send_and_grade(item["question"], item["choices"], item["answerKey"]):
        num_correct += 1
    
    accuracy = (num_correct / total) * 100
    print(f"Running Accuracy: {accuracy:.2f}% ({num_correct}/{total})")

print(f"""
--- FINAL RESULTS ---
Total Questions: {total}
Number Correct: {num_correct}
Final Accuracy: {(num_correct/total*100) if total > 0 else 0:.2f}%
""")