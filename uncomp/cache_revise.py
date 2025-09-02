import torch
from typing import List, Optional, Tuple, Dict, Any
from transformers.cache_utils import DynamicCache

def update_past_key_value(
    past_key_value,
    key_states: tuple,
    value_states: tuple,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
    manager = None,
):
    # Update the cache
    if len(past_key_value.key_cache) <= layer_idx:
        past_key_value.key_cache.append(key_states)
        past_key_value.value_cache.append(value_states)
    else:
        for i in range(manager.head_group_num):
            past_key_value.key_cache[layer_idx][i] = torch.cat([past_key_value.key_cache[layer_idx][i], key_states], dim=-2)
            past_key_value.value_cache[layer_idx][i] = torch.cat([past_key_value.value_cache[layer_idx][i], value_states], dim=-2)
    return past_key_value.key_cache[layer_idx], past_key_value.value_cache[layer_idx]

@classmethod
def from_legacy_cache(cls, 
                      past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                      manager=None,
                      ) -> "DynamicCache":
    """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
    cache = cls()
    if past_key_values is not None:
        for layer_idx in range(len(past_key_values)):
            key_states, value_states = past_key_values[layer_idx]
            update_past_key_value(cache, key_states, value_states, layer_idx, manager)
    return cache

def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    if len(self.key_cache) <= layer_idx:
        return 0
    # return self.key_cache[layer_idx][0].shape[-2]
    return max(self.key_cache[layer_idx][i].shape[-2] for i in range(len(self.key_cache[layer_idx])))