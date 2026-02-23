#!/usr/bin/python
from ollama import chat, ChatResponse, ResponseError
from datasets import load_dataset
import sys,time

ollamaOn=False
thinking=False
model=""
shard_count=10
split="test"
subset="ARC-Challenge"
dataset="allenai/ai2_arc"
sysPrompt="""
Choose the answer which is most correct.
Respond only with the label of the choice in the final answer.
Do not respond with any other text such as 'the correct answer is' or the text of the answer, ONLY the label(A, B, 3, etc.)
"""

if '-o' in sys.argv:
    ollamaOn=True;
    model=sys.argv[1+sys.argv.index("-o")]
if '-s' in sys.argv:
    shard_count=int(sys.argv[1+sys.argv.index("-s")])
if '-t' in sys.argv:
    thinking=True
if '-s' in sys.argv:
    split=sys.argv[1+sys.argv.index("-s")]
if '-ss' in sys.argv:
    subset=sys.argv[1+sys.argv.index("-ss")]
if '-ds' in sys.argv:
    dataset=sys.argv[1+sys.argv.index("-ds")]


def send_and_grade(model: str, question: str, choices: dict[str:list[str]], answer: str):
    q=question+'\n'+'\n'.join(map(': '.join, zip(choices["label"],choices["text"])))
    print(q.strip())
    if ollamaOn:
        response: ChatResponse = chat(model=model,messages=[
            {
                "role": "user",
                "content": sysPrompt+q
            }
        ],think=thinking)
        print(model,"response:",response.message.content.strip())
        if response.message.content.strip().startswith(answer):
            print("correct")
            return True
        else:
            print("incorrect: correct answer was",answer)
    return False

data = load_dataset(dataset,subset,split=split).remove_columns("id").shuffle(time.time_ns()).shard(shard_count,0)
questions = data["question"]
choices = data["choices"]
answers = data["answerKey"]

numcorrect=0
total=0
for (question,choice_list,answer) in zip(questions,choices,answers):
    total+=1
    if send_and_grade(model=model,question=question,choices=choice_list,answer=answer):
        numcorrect+=1
print(f"""Results:
Total Questions: {total}
Number Correct: {numcorrect}({numcorrect/total})""")