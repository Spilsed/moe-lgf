import transformers
from transformers.models.jetmoe.modeling_jetmoe import JetMoeTopKGating
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn

from typing import Type, TypeVar

TModule = TypeVar("TModule", bound=nn.Module)

class StatefulGate(nn.Module):
    def __init__(self, input_size: int = 2048, experts: int = 8):
        super(StatefulGate, self).__init__()
        self.input_size: int = input_size
        self.experts: int = experts

        self.linear = nn.Linear(input_size + experts, experts)

        self.gate_state: torch.Tensor | None = None
    
    def reset_state(self):
        self.gate_state = None
    
    def forward(self, x: torch.Tensor):
        if self.gate_state is None:
            self.gate_state = torch.zeros(x.size(0), self.experts, device=x.device)
        
        combined = torch.cat((x, self.gate_state[:x.shape[0]]), dim=1)
        output = self.linear(combined)

        self.gate_state = output.detach()

        return output

class DeconstructedJetMoE:
    def __init__(self, attn_gating_function: Type[TModule], mlp_gating_function: Type[TModule]):
        self.base = AutoModelForCausalLM.from_pretrained(
            "jetmoe/jetmoe-8b", dtype="auto", device_map="auto")
        self.model: transformers.JetMoeModel = self.base.model
        self.tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b")

        self.attn_gates = [attn_gating_function() for _ in range(24)]
        self.mlp_gates = [mlp_gating_function() for _ in range(24)]

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

            attn_router.layer = self.attn_gates[i]
            mlp_router.layer = self.mlp_gates[i]

    def __str__(self) -> str:
        return ""
