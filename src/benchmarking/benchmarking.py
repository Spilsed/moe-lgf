#!/usr/bin/python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import signal,sys,time,torch

torch.set_num_threads(14)

model=""
num_rows=100
whole_set=False
split="test"
subset="ARC-Challenge"
dataset="allenai/ai2_arc"
sysPrompt="""
Choose the answer which is most correct.
Respond only with the label of the choice in the final answer.
Do not respond with any other text such as 'the correct answer is' or the text of the answer, ONLY the label(A, B, C, 1, 2, 3, etc.)
"""
model="inc/jetmoe-local"
accelerator = Accelerator()
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True, trust_remote_code=True)
llm=AutoModelForCausalLM.from_pretrained(
    model,
    local_files_only=True,
    trust_remote_code=True,
    device_map="auto",
)


if '-c' in sys.argv:
    if (a:=sys.argv[1+sys.argv.index("-c")])=="all":
        whole_set=True
    else:
        num_rows=int(a)
if '-s' in sys.argv:
    split=sys.argv[1+sys.argv.index("-s")]
if '-ss' in sys.argv:
    subset=sys.argv[1+sys.argv.index("-ss")]
if '-ds' in sys.argv:
    dataset=sys.argv[1+sys.argv.index("-ds")]

answer_lookup={"1":"A","2":"B","3":"C","4":"D","A":"1","B":"2","C":"3","D":"4"}

def load_data():
    if dataset=="allenai/ai2_arc":
        return load_dataset(dataset,subset,split=split).remove_columns("id").map(lambda data: {"choices": data["choices"]["label"]+": "+data["choices"]["text"]}).shuffle(time.time_ns())
    elif dataset=="Rowan/hellaswag":
        return load_dataset(dataset,split=split).remove_columns(["ctx_a","ctx_b","ind","source_id"]).rename_column("ctx","question").rename_column("endings","choices").rename_column("label","answerKey").map(lambda data: {"choices": list(map(lambda ix: answer_lookup[str(ix[0]+1)]+": "+ix[1], enumerate(data["choices"]))), "answerKey": str(int(data["answerKey"])+1)}).shuffle(time.time_ns())

def send_and_grade(model: str, question: str, choices: list[str], answer: str):
    q=question+'\n'+'\n'.join(choices)
    print(q.strip())
    content = [
        {"role": "system", "content": sysPrompt},
        {"role": "user", "content": q},
    ]
    start=time.time()
    prompt = tokenizer.apply_chat_template(
        content,
        tokenize=False,
        add_generation_prompt=True
    )
    m_input=tokenizer(prompt,return_tensors='pt').to(accelerator.device)
    t=time.time()
    print(t-start,"Seconds,",input_length:=m_input['input_ids'].shape[-1],"Input Tokens") 
    gen_ids=llm.generate(
        **m_input,
        do_sample=True,
        temperature=.8,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(time.time()-t,"Seconds,",gen_ids.shape[1]-input_length,"Output Tokens")
    response=tokenizer.decode(gen_ids[0][input_length:],skip_special_tokens=True).strip()
    print(model,"response:",response)
    if response.startswith(answer):
        print("correct")
        return True
    elif response[0] in answer_lookup and answer_lookup[response[0]]==answer:
        print("correct with replacement")
        return True
    else:
        print("incorrect: correct answer was",answer)

    return False

if whole_set:
    data = load_data()
    num_rows=sum(1 for _ in data["question"])
else:
    data = load_data().take(num_rows)
questions = data["question"]
choices = data["choices"]
answers = data["answerKey"]

paused=False
def handle_int(signum, frame):
    global paused
    paused=True

signal.signal(signal.SIGINT, handle_int)

numcorrect=0
total=0
for (question,choice_list,answer) in zip(questions,choices,answers):
    if paused:
        if input("Paused. Quit now? y/[n]: ").strip().lower() == 'y':
            break
        else:
            paused = False
            print("Resuming...")
    total+=1
    print("Question",total,"out of",num_rows)
    if send_and_grade(model=model,question=question,choices=choice_list,answer=answer):
        numcorrect+=1
    print("Current score:",numcorrect,"correct out of",total,f"({numcorrect/total*100}%)")

print(f"""Results:
Total Questions: {total}
Number Correct: {numcorrect}({numcorrect/total*100}%)""")