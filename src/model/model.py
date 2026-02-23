import transformers
from transformers.models.jetmoe.modeling_jetmoe import JetMoeTopKGating
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable


class DeconstructedJetMoE:
    attn_gating_function: Callable
    mlp_gating_function: Callable

    def __init__(self, attn_gating_function: Callable, mlp_gating_function: Callable):
        self.model: transformers.JetMoeModel = AutoModelForCausalLM.from_pretrained(
            "jetmoe/jetmoe-8b", dtype="auto", device_map="auto").model
        self.tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b")

        self.assign_function(attn_gating_function, mlp_gating_function)

    def assign_function(self, attn_x: Callable, mlp_x: Callable) -> None:
        modules = self.model.layers._modules
        for i, module in enumerate(modules.values()):
            current_module = modules[str(i)]
            attn_router = current_module._modules[
                'self_attention']._modules['experts']._modules['router']
            mlp_router = current_module._modules['mlp']._modules['router']

            if not isinstance(attn_router, JetMoeTopKGating) or not isinstance(mlp_router, JetMoeTopKGating):
                raise Exception('Router not found in MoE module :(')

            attn_router.layer = attn_x
            mlp_router.layer = mlp_x

        self.attn_gating_function = attn_x
        self.mlp_gating_function = mlp_x

    def __str__(self) -> str:
        return ""
