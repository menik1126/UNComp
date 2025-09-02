from importlib.metadata import version
import transformers

from uncomp.llama_model import LlamaDecoderLayer_forward,llama_attn_forward_Uncomp
from uncomp.mistral_model import MistralDecoderLayer_forward,mistral_attn_forward_Uncomp

from uncomp.llama_model import prepare_inputs_for_generation_llama
from uncomp.mistral_model import prepare_inputs_for_generation_mistral
from functools import wraps, partialmethod

from uncomp.cache_revise import from_legacy_cache,get_seq_length

def replace_llama(method,manager):
    print("Using method: ",method)
    if method == "fullkv":
        print("Using fullkv!")
        pass
    else :
        transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = partialmethod(
                    LlamaDecoderLayer_forward, manager = manager,
                )
         
        transformers.models.llama.modeling_llama.LlamaAttention.forward = partialmethod(
                llama_attn_forward_Uncomp, manager = manager,   
            )   

        if manager.method_name in manager.head_granularity  and manager.method_name not in manager.delet_head_set:
            transformers.cache_utils.DynamicCache.from_legacy_cache = partialmethod(
                        from_legacy_cache, manager = manager,
                    )
            transformers.cache_utils.DynamicCache.get_seq_length = get_seq_length
         
    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama



def replace_mistral(method,manager):
    print("Using method: ",method)
    if method == "fullkv":
        print("Using fullkv!")
        pass
    else:
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = partialmethod(
                    mistral_attn_forward_Uncomp, manager = manager,
                ) 
        transformers.models.mistral.modeling_mistral.MistralDecoderLayer.forward = partialmethod(
                    MistralDecoderLayer_forward, manager = manager,
        )
        
        if manager.method_name in manager.head_granularity and manager.method_name not in manager.delet_head_set:
            transformers.cache_utils.DynamicCache.from_legacy_cache = partialmethod(
                        from_legacy_cache, manager = manager,
                    )
            transformers.cache_utils.DynamicCache.get_seq_length = get_seq_length    
        
    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral
