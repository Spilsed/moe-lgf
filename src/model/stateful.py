import torch
from torch import FloatTensor, LongTensor, Tensor, nn
from transformers import Cache, JetMoeConfig
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models import JetMoeModel
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

class StatefulGate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StatefulGate, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.gate_layer = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.state = None

    def reset_state(self):
        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        combined = torch.cat((x, self.state), dim=1)
        
        gate_output = self.sigmoid(self.gate_layer(combined))
        
        self.state = gate_output.detach()
        
        return gate_output

class StatefulJetMoeModel(JetMoeModel):
    def __init__(self, config: JetMoeConfig = JetMoeConfig()):
        super().__init__(config)
        self.stateful_gate = StatefulGate(config.hidden_size, config.hidden_size)
    
    def forward(
        self,
        input_ids: LongTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs]
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if inputs_embeds is None:
                raise Exception("Input embeds count not be set")
        
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            self.stateful_gate.reset_state()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.LongTensor(torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ))
        
        if position_ids is None:
            position_ids = torch.LongTensor(cache_position.unsqueeze(0))

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )