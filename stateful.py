from src.model.stateful import StatefulJetMoeModel
import torch
from transformers import AutoTokenizer
from transformers.models import JetMoeForCausalLM, JetMoeConfig

model = StatefulJetMoeModel()
causal = JetMoeForCausalLM(JetMoeConfig())
tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b-chat")

output = model.forward(torch.LongTensor([[1, 22557, 28725, 28705]]), attention_mask=torch.ones((1, 4))).last_hidden_state

if output == None:
    raise Exception("Cannot get last hidden state!")

logits = causal.lm_head(output[:, -1, :])
print(logits)
print(logits.shape)

next_token_id = torch.argmax(logits, dim=-1)
print(next_token_id)
print(next_token_id.shape)

print(tokenizer.decode(next_token_id))