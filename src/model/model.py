import transformers
from transformers.models.jetmoe.modeling_jetmoe import JetMoeTopKGating
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from torch import nn

from typing import Type, TypeVar

TModule = TypeVar("TModule", bound=nn.Module)

class StatefulGate(nn.Module):
    def __init__(self, input_size: int = 2048, experts: int = 8):
        super(StatefulGate, self).__init__()
        self.input_size = input_size
        self.experts = experts
        self.linear = nn.Linear(input_size + experts, experts, bias=False)
        
        self.register_buffer("gate_state", torch.zeros(1, experts))
        self.has_initialized = False

    def reset_state(self):
        self.gate_state.zero_()
        self.has_initialized = False
    
    def forward(self, x: torch.Tensor):
        seq_len = x.size(0)

        if not self.has_initialized:
            self.gate_state = torch.zeros(1, self.experts, device=x.device)
            self.has_initialized = True
        
        state_input = self.gate_state.expand(seq_len, -1)

        combined = torch.cat((x, state_input), dim=1)
        output = self.linear(combined)

        self.gate_state = output[-1:].detach()

        return output

class DeconstructedJetMoE:
    def __init__(self, attn_gating_function: Type[TModule], mlp_gating_function: Type[TModule]):
        self.base = AutoModelForCausalLM.from_pretrained(
            "/home/titan/Downloads/moe-lgf/inc/jetmoe-local", local_files_only=True, dtype="auto", device_map="auto")
        self.model: transformers.JetMoeModel = self.base.model
        self.tokenizer = AutoTokenizer.from_pretrained("/home/titan/Downloads/moe-lgf/inc/jetmoe-local", local_files_only=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.attn_gates = [attn_gating_function().to(device) for _ in range(24)]
        self.mlp_gates = [mlp_gating_function().to(device) for _ in range(24)]

        self.assign_functions()

    def generate(self, text: str) -> None:
        if self.tokenizer is None:
            raise Exception('Tokenizer not found!')

        outputs = self.base.generate(
            **self.tokenizer(text, return_tensors="pt"),
            max_new_tokens=16,
            do_sample=True,
            temperature=0.9,
            top_p=0.9
        )

        return self.tokenizer.decode(outputs)[0]

    def assign_functions(self) -> None:
        modules = self.model.layers._modules
        for i, _ in enumerate(modules.values()):
            current_module = modules[str(i)]
            attn_router = current_module._modules['self_attention']._modules['experts']._modules['router']
            mlp_router = current_module._modules['mlp']._modules['router']

            if not isinstance(attn_router, JetMoeTopKGating) or not isinstance(mlp_router, JetMoeTopKGating):
                raise Exception('Router not found in MoE module :(')

            original_weights = attn_router.layer.weight.data
            with torch.no_grad():
                self.attn_gates[i].linear.weight[:, :2048] = original_weights
                self.attn_gates[i].linear.weight[:, 2048:] = 0.0 
            
            original_weights = mlp_router.layer.weight.data
            with torch.no_grad():
                self.mlp_gates[i].linear.weight[:, :2048] = original_weights
                self.mlp_gates[i].linear.weight[:, 2048:] = 0.0 

            attn_router.layer = self.attn_gates[i]
            mlp_router.layer = self.mlp_gates[i]

    def __str__(self) -> str:
        return ""

if __name__ == "__main__":
    model = DeconstructedJetMoE(StatefulGate, StatefulGate)

    print(model.generate("Hello, "))