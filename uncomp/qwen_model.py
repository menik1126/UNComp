import math
from abc import ABC
from typing import Optional, Tuple, Dict, Union, List
import torch.nn.functional as F
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM,Qwen2Model,Qwen2DecoderLayer,Qwen2MLP,\
    Qwen2RMSNorm,Qwen2FlashAttention2,apply_rotary_pos_emb,repeat_kv
from transformers.cache_utils import Cache,DynamicCache,HybridCache
# from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal
from transformers.modeling_flash_attention_utils import _upad_input, flash_attn_varlen_func, flash_attn_func
from my_utils.augment_index import aug_indices_comb_classification
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
import os
import copy
import inspect
import logging
from torch.nn import CrossEntropyLoss
from utils.logger import Logger
logger = Logger()
logger.set_console_level(logging.DEBUG)
from utils.entropy_utils import (
    get_head_pattern_attn_entropy,
)

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_varlen_func

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1",
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
        causal = is_causal and query_length != 1
    
    
    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if is_flash_attn_greater_or_equal("2.4.1"):
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            # return_attn_probs = True,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, 
            # return_attn_probs = True,
            **flash_kwargs
        )
        
    return attn_output
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_new(q, k, cos_query, sin_query, cos_key,sin_key, position_ids=None, unsqueeze_dim=1):
    cos_query = cos_query.unsqueeze(unsqueeze_dim)
    sin_query = sin_query.unsqueeze(unsqueeze_dim)
    cos_key = cos_key.unsqueeze(unsqueeze_dim)
    sin_key = sin_key.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_query) + (rotate_half(q) * sin_query)
    k_embed = (k * cos_key) + (rotate_half(k) * sin_key)
    return q_embed, k_embed

THRES = float(os.environ.get("THRES", 0.1))
RECENT_TOKENS = int(os.environ.get("RECENT_TOKENS", 1))
class Manager:
    last_attn = None
    pass
class Qwen2FlashAttention2PCW(Qwen2FlashAttention2):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None,
                capacity = 512, n_windows=1, kv_cache_eviction=False,
                kv_cache_dynamic=False,stage_eviction=False,
                calibration_mode=0,calibration_stage=None,
                head_datas=None,manager=None,
                 ):
        super().__init__(config, layer_idx)
        self.window_size = 8
        if head_datas is not None:
            self.head_datas = torch.tensor(head_datas)
            self.head_indices1 = self.head_datas[self.layer_idx]
            self.bsz = 1
            num_attention_heads = self.config.num_attention_heads
            self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,num_attention_heads//2,-1),torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,num_attention_heads//2,-1)]
        
        self.capacity = capacity
        self.eviction_len = 0
        self.manager = manager
        self.kv_cache_eviction_prefill = kv_cache_eviction
        self.kv_cache_dynamic = kv_cache_dynamic
        self.query_len = self.window_size
        self.n_windows = n_windows
        self.stage_eviction = stage_eviction
        self.kernel_size = 7
        self.pooling = 'maxpool'
        self.calibration_mode = calibration_mode
        self.calibration_stage=calibration_stage
    
    def get_head_type(self, key_states, query_states):
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(key_states.size(-1))
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        select_topk = query_states.shape[1]//2
        svdn = 32 
        head_pattern = get_head_pattern_attn_entropy(attn_weights,query_states,0,svdn,0,[0,select_topk])
        svdn = "svd" + str(svdn)
        filename = f"./qwen2/{svdn}/"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        filename = filename + "head_type_search_layer" + str(self.layer_idx) + ".csv"
        mode = 'a'
        with open(filename, mode, newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            writer.writerow(head_pattern.to(torch.int8).tolist())
        if self.layer_idx == self.config.num_hidden_layers-1:
            print(self.layer_idx)
            # assert 1==0
   
    def _update_kv_second_prefill(self,past_key_value, key_states, query_states, value_states, q_len):
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
        self.query_len = self.window_size
        # print("attn_weights_sum.shape:{}".format(attn_weights_sum.shape))
        # assert 1==0
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        #print("self.capacity0:{}".format(self.capacity))
        #assert 1==0
        capacity = min(self.capacity, q_len)
        top_k = capacity*self.n_windows-self.query_len
        # logger.debug(f"top_k:{top_k}")
        # logger.debug(f"capacity:{capacity}")
        # logger.debug(f"self.query_len:{self.query_len}")
        # logger.debug(f"attn_weights_sum.shape: {attn_weights_sum.shape}")
        self.new_capacity = capacity*self.n_windows
        indices = attn_cache.topk(top_k, dim=-1).indices
        indices = indices.sort(dim=-1).values
        indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  
        indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
        attn_weights_compress = attn_weights[:, :, -self.window_size:, :-self.query_len].gather(dim = -1, index = indices_attn)
        k_past_compress = key_states[:, :, :-self.query_len, :].gather(dim = 2, index = indices_expanded)
        v_past_compress = value_states[:, :, :-self.query_len, :].gather(dim = 2, index = indices_expanded)
        k_cur = key_states[:, :, -self.query_len:, :]
        v_cur = value_states[:, :, -self.query_len:, :]
        attn_weights_cur = attn_weights[:, :, -self.window_size:, -self.query_len:]
        
        key_states_compress = torch.cat([k_past_compress, k_cur], dim = 2).contiguous()
        value_states_compress = torch.cat([v_past_compress, v_cur], dim = 2).contiguous()
        self.attn_weights = torch.cat([attn_weights_compress,attn_weights_cur],dim=-1)
        # logger.debug(f"self.attn_weights.shape:{self.attn_weights.shape}")
        
        del k_past_compress, k_cur, v_past_compress, v_cur, key_states, value_states, attn_weights_compress, attn_weights_cur
        torch.cuda.empty_cache()
        
        past_key_value.key_cache[self.layer_idx] = key_states_compress
        past_key_value.value_cache[self.layer_idx] = value_states_compress

    def _update_kv_first_prefill(self,past_key_value, key_states, query_states, value_states, q_len):
        token_prompt_size = self.token_prompt_size
        
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, token_prompt_size : -self.query_len].sum(dim = -2)
        # print("attn_weights_sum.shape:{}".format(attn_weights_sum.shape))
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        
        #print("self.capacity0:{}".format(self.capacity))
        #assert 1==0
        capacity = min(self.capacity, q_len)
        # logger.info("capacity:{}".format(capacity))
        top_k = capacity-self.query_len
        # logger.info("top_k:{}".format(top_k))
        # logger.info("attn_cache.shape:{}".format(attn_cache.shape))
        indices = attn_cache.topk(top_k, dim=-1).indices
        indices = indices.sort(dim=-1).values
        indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  
        indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
        attn_weights_compress = attn_weights[:, :, -self.window_size:, token_prompt_size:-self.query_len].gather(dim = -1, index = indices_attn)
        k_past_compress = key_states[:, :, token_prompt_size:-self.query_len, :].gather(dim = 2, index = indices_expanded)
        v_past_compress = value_states[:, :, token_prompt_size:-self.query_len, :].gather(dim = 2, index = indices_expanded)
        k_cur = key_states[:, :, -self.query_len:, :]
        v_cur = value_states[:, :, -self.query_len:, :]

        key_states_compress = torch.cat([k_past_compress, k_cur], dim = 2).contiguous()
        value_states_compress = torch.cat([v_past_compress, v_cur], dim = 2).contiguous()

        del k_past_compress, k_cur, v_past_compress, v_cur, key_states, value_states, attn_weights_compress
        torch.cuda.empty_cache()
        
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx,cache_kwargs=None)
    
    def update_past_key_value(
        self,
        past_key_value,
        key_states: tuple,
        value_states: tuple,
        layer_idx: int,
        mode: int,
    ):  
        
        if len(past_key_value.key_cache) <= layer_idx:
            past_key_value.key_cache.append(key_states)
            past_key_value.value_cache.append(value_states)
        else:
            if mode == 1:
                past_key_value.key_cache[layer_idx] = torch.cat([past_key_value.key_cache[layer_idx], key_states[:,self.head_pattern,:,:]], dim=-2)
                past_key_value.value_cache[layer_idx] = torch.cat([past_key_value.value_cache[layer_idx], value_states[:,self.head_pattern,:,:]], dim=-2)
            else:
                groups_num = len(past_key_value.key_cache[layer_idx])
                for i in range(groups_num):
                    past_key_value.key_cache[layer_idx][i] = torch.cat([past_key_value.key_cache[layer_idx][i], key_states[:,self.head_pattern[i],:,:]], dim=-2)
                    past_key_value.value_cache[layer_idx][i] = torch.cat([past_key_value.value_cache[layer_idx][i], value_states[:,self.head_pattern[i],:,:]], dim=-2)
        return past_key_value.key_cache[layer_idx], past_key_value.value_cache[layer_idx]

    def _update_kv_first_prefill_uncomp(self,past_key_value, key_states, query_states, value_states, q_len):
        token_prompt_size = self.token_prompt_size
        query_len = self.query_len
        
        num_hidden_layers = self.config.num_hidden_layers
        num_attention_heads = query_states.shape[1]
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(key_states.size(-1))
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        bsz, num_heads, q_len, head_dim = query_states.shape
        max_capacity_prompt = min(self.capacity, q_len)
        
        if self.window_size > q_len:
            self.window_size = q_len
        if self.window_size > max_capacity_prompt//2:
            self.window_size = max_capacity_prompt//2
            self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,num_attention_heads//2,-1),torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers//2,-1)]
        
        # max_capacity_prompt = (max_capacity_prompt-query_len) * 3 // 2 
        # max_capacity_prompts = [max_capacity_prompt//2+query_len,max_capacity_prompt+query_len]
        new_max_capacity_prompt = max_capacity_prompt - self.query_len
        max_capacity_prompt_1 = new_max_capacity_prompt-new_max_capacity_prompt//3
        max_capacity_prompt_2 = new_max_capacity_prompt*2-max_capacity_prompt_1
        max_capacity_prompts = [max_capacity_prompt_2,max_capacity_prompt_1]
        
        if q_len <= max_capacity_prompt:
            max_capacity_prompts = [q_len,q_len]
        max_capacity_prompts = [min(prompt, q_len-self.query_len) for prompt in max_capacity_prompts]
        if self.layer_idx == num_hidden_layers-1:
            # print("query_len",token_prompt_size)
            # print("self.capacity",self.capacity)
            # print("max_capacity_prompt",max_capacity_prompt)
            print("max_capacity_prompts",max_capacity_prompts)
        attn_weights = attn_weights
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.query_len].sum(dim = -2)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        top_k = max_capacity_prompts[1]
        indices = attn_cache.topk(top_k, dim=-1).indices
        indices1 = indices.sort(dim=-1).values
        for i in range(len(self.recent_indices_generate)):
            self.recent_indices_generate[i] = torch.arange(-self.query_len,0,device='cuda').view(1, 1, -1).expand(self.bsz,num_attention_heads//2,-1).to(indices1.device)
            # self.recent_indices_generate[i] = self.recent_indices_generate[i]
        self.head_indices1 = self.head_indices1.to(indices1.device)
        recent_indices = self.recent_indices_generate[0]+q_len
        num_heads = num_attention_heads // 2
        indices_1 = torch.cat([indices1[:,self.head_indices1[-num_heads:],:],recent_indices],dim=-1)
        indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        revise_key_states=key_states[:,self.head_indices1[-num_heads:],:,:].gather(dim = 2, index = indices_expanded)
        revise_value_states=value_states[:,self.head_indices1[-num_heads:],:,:].gather(dim = 2, index = indices_expanded)
        
        max_capacity_prompt_2 = max_capacity_prompts[0]
        top_k2 = max_capacity_prompt_2
        indices = attn_cache.topk(top_k2, dim=-1).indices
        indices_2 = indices.sort(dim=-1).values
        indices_2 = torch.cat([indices_2[:,self.head_indices1[:num_heads],:],recent_indices],dim=-1)
        indices_expanded_2  = indices_2.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        revise_value_states_2 = value_states[:,self.head_indices1[:num_heads],:,:].gather(dim=2,index=indices_expanded_2)
        revise_key_states_2 = key_states[:,self.head_indices1[:num_heads],:,:].gather(dim=2, index=indices_expanded_2)
        key1 = revise_key_states
        key2 = revise_key_states_2
        value1 = revise_value_states
        value2 = revise_value_states_2
        revise_key_states = [key1, key2]
        revise_value_states = [value1, value2]
        self.head_pattern = [self.head_indices1[-num_heads:], self.head_indices1[:num_heads]]

        del key_states, value_states,attn_weights_sum
        torch.cuda.empty_cache()
        self.update_past_key_value(past_key_value, revise_key_states, revise_value_states, self.layer_idx, mode=1)
        
    def _update_kv_first_prefill_h2o(self,past_key_value, key_states, query_states, value_states, q_len):
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(key_states.size(-1))
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
        
        bsz, num_heads, q_len, head_dim = query_states.shape
        capacity = min(self.capacity, q_len)
        recent_size = self.recent_size = capacity*508//512
        hh_size = self.hh_size = capacity*4//512
        if hh_size==0:
            hh_size = self.hh_size = 1
            recent_size = self.recent_size = capacity - hh_size
        if q_len <= recent_size+hh_size:
            recent_size = self.recent_size = q_len - hh_size
        self.cache_size = recent_size + hh_size
        attn_weights_sum = attn_weights.sum(0).sum(1)
        select_hh_scores = attn_weights_sum[:, :q_len - recent_size]
        _, keep_topk = torch.topk(select_hh_scores, hh_size, dim=-1)
        keep_topk = keep_topk.sort().values 
        keep_recent = torch.arange(q_len - recent_size, q_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)
        mask = torch.zeros(attn_weights_sum.shape, dtype=torch.bool).to(attn_weights.device)
        mask = mask.scatter(-1, keep_idx, 1)
        key_states_compress = key_states.squeeze(0)[mask].view(bsz, num_heads, -1, head_dim)
        value_states_compress = value_states.squeeze(0)[mask].view(bsz, num_heads, -1, head_dim)
        del key_states, value_states,attn_weights_sum
        torch.cuda.empty_cache()
        
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx,cache_kwargs=None)
    
    def _update_kv_first_prefill_streamingllm(self,past_key_value, key_states, query_states, value_states, q_len):
        bsz, num_heads, q_len, head_dim = query_states.shape
        max_capacity_prompt = min(self.capacity, q_len)
        self.window_size = max_capacity_prompt // 7
        
        indices = torch.tensor(range(max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
        indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        key_states_compress = torch.cat([k_past_compress, k_cur], dim = 2)
        value_states_compress = torch.cat([v_past_compress, v_cur], dim = 2)
    
        del k_past_compress, k_cur, v_past_compress, v_cur, key_states, value_states
        torch.cuda.empty_cache()
        
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx,cache_kwargs=None)
    
    def _update_kv_first_prefill_pyramidkv(self,past_key_value, key_states, query_states, value_states, q_len):
        bsz, num_heads, q_len, head_dim = query_states.shape
        capacity = min(self.capacity, q_len)
        self.beta = 20
        my_max = int(capacity * 1.5)
        min_num = capacity // self.beta
        max_num = my_max - min_num
        if max_num >= q_len:
            max_num = q_len
            min_num = my_max - max_num
        steps = (max_num - min_num) // self.config.num_hidden_layers
        max_capacity_prompt = min_num + (self.config.num_hidden_layers-self.layer_idx) * steps
        if q_len < max_capacity_prompt:
            max_capacity_prompt = q_len
        if max_capacity_prompt < self.window_size:
            max_capacity_prompt = self.window_size
        self.cache_size = max_capacity_prompt
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # print("attn_weights.shape",attn_weights.shape)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        top_k = max_capacity_prompt - self.window_size
        indices = attn_cache.topk(top_k, dim=-1).indices
        indices = indices.sort(dim=-1).values
        indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
        attn_weights_compress = attn_weights[:, :, -self.window_size:, :-self.window_size].gather(dim = -1, index = indices_attn)
        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices_expanded)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices_expanded)
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        key_states_compress = torch.cat([k_past_compress, k_cur], dim = 2)
        value_states_compress = torch.cat([v_past_compress, v_cur], dim = 2)
       
        del k_past_compress, k_cur, v_past_compress, v_cur, key_states, value_states, attn_weights_compress
        torch.cuda.empty_cache()
        
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx,cache_kwargs=None)
    
    def _update_kv_first_prefill_snapkv(self,past_key_value, key_states, query_states, value_states, q_len):
        bsz, num_heads, q_len, head_dim = query_states.shape
        capacity = min(self.capacity, q_len)
        
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        top_k = capacity - self.window_size
        indices = attn_cache.topk(top_k, dim=-1).indices
        indices = indices.sort(dim=-1).values
        indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
        attn_weights_compress = attn_weights[:, :, -self.window_size:, :-self.window_size].gather(dim = -1, index = indices_attn)
        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices_expanded)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices_expanded)
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        key_states_compress = torch.cat([k_past_compress, k_cur], dim = 2)
        value_states_compress = torch.cat([v_past_compress, v_cur], dim = 2)
       
        del k_past_compress, k_cur, v_past_compress, v_cur, key_states, value_states, attn_weights_compress
        torch.cuda.empty_cache()
        
        past_key_value.update(key_states_compress, value_states_compress, self.layer_idx,cache_kwargs=None)
        
    def _update_kv_input(self, attn_weights, kv_seq_len):
        self.new_capacity = attn_weights.shape[-1]
        self.window_size = min(self.window_size,kv_seq_len)
        self.attn_weights = attn_weights[:,:,-self.window_size:,:]
    
    def _update_kv_generate(self, attn_weights, past_key_value,key_states,value_states,layer_idx):
        _,_,_,head_dim = key_states.shape
        attn_weights_dynamic = self.attn_weights
        # logger.debug(f"attn_weights.shape:{attn_weights.shape}") 
        now_attn_weights = torch.zeros((attn_weights_dynamic.shape[0],attn_weights_dynamic.shape[1],attn_weights_dynamic.shape[2]+1,attn_weights_dynamic.shape[3]+1),device=attn_weights_dynamic.device)
        now_attn_weights[:,:,:-1,:-1] = attn_weights_dynamic
        now_attn_weights[:,:,:,-1] = 0
        now_attn_weights[:,:,-1:,:] = attn_weights
        attn_weights_sum = now_attn_weights[...,-self.window_size:,:-self.window_size].sum(-2)
        cache_size = self.new_capacity
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        indices  = attn_cache.topk(cache_size-self.window_size, dim=-1).indices
        indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
        attn_weights_compress = now_attn_weights[:, :, -self.window_size:, :-self.window_size].gather(dim = -1, index = indices_attn)
        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices_expanded)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices_expanded)
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        attn_weights_cur = now_attn_weights[:, :, -self.window_size:, -self.window_size:]
        revise_key_states = torch.cat([k_past_compress, k_cur], dim = 2)
        revise_value_states = torch.cat([v_past_compress, v_cur], dim = 2)
        self.attn_weights = torch.cat([attn_weights_compress,attn_weights_cur],dim=-1)
        
        past_key_value.key_cache[layer_idx] = revise_key_states
        past_key_value.value_cache[layer_idx] = revise_value_states
    
    def get_attn(self,query_states, key_states):
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :].sum(dim = -2)
        # print("attn_weights_sum.shape:{}".format(attn_weights_sum.shape))
        # if self.pooling == 'avgpool':
        #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        # elif self.pooling == 'maxpool':
        #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        # else:
        #     raise ValueError('Pooling method not supported')

        return attn_weights_sum
    
    def draw_attn(self,query_states, key_states):
        if self.window_size == 0:
            self.window_size = query_states.shape[-2]
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights_sum = attn_weights[:, :, -self.window_size:, :].sum(dim = -2)

        return attn_weights
    
    def eviction_tokens(self,key_states, value_states, attn_weights,layer_idx,threshold, recent_tokens,calibration_stage):
        parts = calibration_stage.split('_')
        last_two_numbers = list(map(int, parts[-2:]))
        if last_two_numbers[0] <= layer_idx <= last_two_numbers[1]:
            attn_sum = attn_weights[:,:,-recent_tokens:,:].sum(0).sum(0).sum(0) / (attn_weights.shape[1]*recent_tokens)
            if torch.sum(attn_weights[:,:,-recent_tokens:,:]) != 32*recent_tokens:
                print(f"torch.sum(attn_sum) : {torch.sum(attn_weights[:,:,-recent_tokens:,:])}")
                raise ValueError("Attention sum is not equal to 32*recent_tokens")
            shape = attn_sum.shape[0]
            device = attn_sum.device
            indices = torch.nonzero(torch.where( attn_sum > threshold, 1, 0))
            indices = indices.squeeze(-1)
            reserved_indices = torch.arange(shape, device=device)
            indices = torch.tensor(list(set(reserved_indices.tolist())-set(indices.tolist())),device=device)
            
            new_key_states = key_states[:,:,indices,:]
            new_value_states = value_states[:,:,indices,:]
        else:
            new_key_states = key_states
            new_value_states = value_states
        return new_key_states, new_value_states
    
    def eviction_tokens_head(self,key_states, value_states, attn_weights,layer_idx,threshold, recent_tokens,calibration_stage,pattern):
        parts = calibration_stage.split('_')
        last_two_numbers = list(map(int, parts[-2:]))
        if last_two_numbers[0] <= layer_idx <= last_two_numbers[1]:
            if recent_tokens == 0:
                recent_tokens = attn_weights.shape[-2]
            attn_sum = attn_weights[:,:,-recent_tokens:,:].sum(0).sum(1) / (attn_weights.shape[0]*recent_tokens)
            all_indices = []
            for i in range(attn_sum.shape[0]):
                indices = torch.nonzero(torch.where( attn_sum[i] > threshold, 1, 0)).squeeze(-1)
                if pattern == "recent":  # evict recent
                    filtered_indices = indices[indices > attn_sum.shape[1]-100]
                elif pattern == "sink": # evict sink
                    filtered_indices = indices[indices < 100]
                elif pattern =="middle" : # evict middel
                    indices_temp = indices[indices < attn_sum.shape[1]-100] 
                    filtered_indices = indices_temp[ indices_temp > 100]
                elif pattern == "all": # evict all
                    filtered_indices = indices
                elif pattern == "recent_sink": # evict recent and sink
                    indices_recent = indices[indices > attn_sum.shape[1]-100] 
                    indices_sink = indices[ indices < 100]
                    filtered_indices = torch.cat([indices_recent,indices_sink],dim=-1)
                all_indices.append(filtered_indices)
            len_indices = [indices.shape[0] for indices in all_indices]
            max_len = max(len_indices)
            len_supplement_indices = [max_len - len_index for len_index in len_indices]
            for i in range(attn_sum.shape[0]):
                attn_now = attn_sum[i] 
                supplement_indices = attn_now.topk(len_supplement_indices[i], dim=-1,largest=False ).indices
                all_indices[i] = torch.cat([all_indices[i],supplement_indices],dim=-1)
            all_indices = torch.stack(all_indices)
            full_indices = torch.arange(key_states.shape[2], device=key_states.device).unsqueeze(0).expand(attn_sum.shape[0],-1)
            mask = torch.ones_like(full_indices)
            mask.scatter_(1, all_indices, 0)
            mask_bool = mask.bool()
            select_indices = full_indices[mask_bool].reshape(attn_sum.shape[0],-1)
            new_key_states = key_states.gather(dim=2,index=select_indices.unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,key_states.shape[-1]))
            new_value_states = value_states.gather(dim=2,index=select_indices.unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,value_states.shape[-1]))
        else:
            new_key_states = key_states
            new_value_states = value_states
        return new_key_states, new_value_states
    
    def eviction_tokens_first_prefill_head(self,key_states, value_states, attn_weights,layer_idx,threshold, recent_tokens,calibration_stage,pattern):
        token_prompt_size = self.token_prompt_size
        query_len = self.query_len
        parts = calibration_stage.split('_')
        last_two_numbers = list(map(int, parts[-2:]))
        if last_two_numbers[0] <= layer_idx <= last_two_numbers[1]:
            if recent_tokens == 0:
                recent_tokens = attn_weights.shape[-2]
            attn_sum = attn_weights[:,:,-recent_tokens:,token_prompt_size:-query_len].sum(0).sum(1) / (attn_weights.shape[0]*recent_tokens)
            all_indices = []
            for i in range(attn_sum.shape[0]):
                indices = torch.nonzero(torch.where( attn_sum[i] > threshold, 1, 0)).squeeze(-1)
                indices += token_prompt_size
                if pattern == "recent":  # evict recent
                    filtered_indices = indices[indices > attn_sum.shape[1]-100]
                elif pattern == "sink": # evict sink
                    filtered_indices = indices[indices < 100]
                elif pattern =="middle" : # evict middle
                    indices_temp = indices[indices < attn_sum.shape[1]-100] 
                    filtered_indices = indices_temp[ indices_temp > 100]
                elif pattern == "all": #  evict all
                    filtered_indices = indices
                all_indices.append(filtered_indices)
            len_indices = [indices.shape[0] for indices in all_indices]
            max_len = max(len_indices)
            self.eviction_len = max_len
            
            len_supplement_indices = [max_len - len_index for len_index in len_indices]
            for i in range(attn_sum.shape[0]):
                attn_now = attn_sum[i] 
                supplement_indices = attn_now.topk(len_supplement_indices[i], dim=-1,largest=False ).indices
                supplement_indices += token_prompt_size
                all_indices[i] = torch.cat([all_indices[i],supplement_indices],dim=-1)
            all_indices = torch.stack(all_indices)
            full_indices = torch.arange(key_states.shape[2], device=key_states.device).unsqueeze(0).expand(attn_sum.shape[0],-1)
            mask = torch.ones_like(full_indices)
            mask.scatter_(1, all_indices, 0)
            mask_bool = mask.bool()
            select_indices = full_indices[mask_bool].reshape(attn_sum.shape[0],-1)
            new_key_states = key_states.gather(dim=2,index=select_indices.unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,key_states.shape[-1]))
            new_value_states = value_states.gather(dim=2,index=select_indices.unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,value_states.shape[-1]))
        else:
            new_key_states = key_states
            new_value_states = value_states
        return new_key_states, new_value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if not isinstance(past_key_value,List):
            raw_len_past_key_value = len(past_key_value.key_cache)
        else:
            raw_len_past_key_value = self.layer_idx
        self.revise = False
        if "uncomp_stage" in self.calibration_stage and raw_len_past_key_value!=self.config.num_hidden_layers:
            manager = self.manager
            num_hidden_layers = self.config.num_hidden_layers
            if self.layer_idx != 0 :
                self.revise = True
                if "long" in self.calibration_stage:
                    min_len = 4000
                else:
                    min_len = 1000
                if "8192" in self.calibration_stage:
                    max_len = 8192-100
                else:
                    max_len = 4096-100       
                steps = (max_len-min_len)//(num_hidden_layers-1)
                keep_seq_len = min_len+steps*(num_hidden_layers-1-self.layer_idx)
                attn_sum = manager.last_attn
                
                if keep_seq_len > hidden_states.shape[-2]:
                    keep_seq_len = hidden_states.shape[-2]
                indices = attn_sum.topk(keep_seq_len,dim=-1).indices
                indices = indices.sort(dim=-1).values
                self.select_indices = indices
                raw_len = position_ids.size(-1)
                position_ids = torch.arange(0,indices.size(-1),device=indices.device).unsqueeze(0)
                position_ids = manager.last_position_ids.gather(-1,indices.unsqueeze(0))
                hidden_states = hidden_states.gather(-2,indices.unsqueeze(-1).unsqueeze(0).expand(hidden_states.size(0),-1,hidden_states.size(-1)))
            manager.last_position_ids = position_ids
            
        bsz, q_len, _ = hidden_states.size()
        output_attentions = False
        # first prefill
        if raw_len_past_key_value!=self.config.num_hidden_layers:
            stage = 0 
        # second prefill
        elif raw_len_past_key_value==self.config.num_hidden_layers and q_len != 1:
            stage = 1 
        # generate
        elif raw_len_past_key_value==self.config.num_hidden_layers and q_len == 1:
            stage = 2 
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        before_rope_key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = before_rope_key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        
        query_states, key_states = apply_rotary_pos_emb(query_states, before_rope_key_states, cos, sin, position_ids)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            
            if raw_len_past_key_value == self.config.num_hidden_layers: # second prefill & generate
                if isinstance(past_key_value[0][0], list):
                    key_states, value_states = self.update_past_key_value(past_key_value,key_states, value_states, self.layer_idx,0)
                else:
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                if self.calibration_stage == "search":
                    self.get_head_type(key_states,query_states)
            raw_query_states = query_states
            raw_key_states = key_states
            raw_value_states = value_states

        # calibration
        threshold = THRES
        recent_tokens = RECENT_TOKENS
        if recent_tokens == 0:
            self.window_size = q_len
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        
        if raw_len_past_key_value == self.config.num_hidden_layers and isinstance(past_key_value[0][0], list):
            attn_outputs = []
            for i, (key_state,value_state) in enumerate(zip(key_states,value_states)):
                query_state = query_states[:,self.head_pattern[i],:,:].transpose(1, 2)
                key_state = key_state.transpose(1, 2)
                value_state = value_state.transpose(1, 2)

                dropout_rate = self.attention_dropout if self.training else 0.0
                
                input_dtype = query_state.dtype
                if input_dtype == torch.float32:
                    if torch.is_autocast_enabled():
                        target_dtype = torch.get_autocast_gpu_dtype()
                    # Handle the case where the model is quantized
                    elif hasattr(self.config, "_pre_quantization_dtype"):
                        target_dtype = self.config._pre_quantization_dtype
                    else:
                        target_dtype = self.q_proj.weight.dtype

                    query_state = query_state.to(target_dtype)
                    key_state = key_state.to(target_dtype)
                    value_state = value_state.to(target_dtype)
                if (
                    self.config.use_sliding_window
                    and getattr(self.config, "sliding_window", None) is not None
                    and self.layer_idx >= self.config.max_window_layers
                ):
                    sliding_window = self.config.sliding_window
                else:
                    sliding_window = None

                attn_output = _flash_attention_forward(
                    query_state,
                    key_state,
                    value_state,
                    attention_mask,
                    q_len,
                    dropout=dropout_rate,
                    sliding_window=getattr(self, "sliding_window", None),
                    use_top_left_mask=self._flash_attn_uses_top_left_mask,
                    is_causal=self.is_causal,
                )
                attn_outputs.append(attn_output)    
            attn_output = torch.cat(attn_outputs,dim=2)
            for i in range(len(key_states)):
                if len(self.head_pattern[i]) != 0:
                    attn_output[:,:,self.head_pattern[i],:] = attn_outputs[i]
        else:
            if self.calibration_mode != 0:
                attn_weights = torch.matmul(query_states[:,:,-recent_tokens:,:], raw_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, -recent_tokens:, : raw_key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            if self.calibration_mode == 1:
                # second prefill calibration
                if "prefill_2" in self.calibration_stage and stage==1: 
                    if "head" in self.calibration_stage:
                        if "head_recent" in self.calibration_stage:
                            pattern = "recent"
                        elif "head_sink" in self.calibration_stage:
                            pattern = "sink"
                        elif "head_middle" in self.calibration_stage:
                            pattern = "middle"
                        elif "head_all" in self.calibration_stage:
                            pattern = "all"
                        elif "head_recent_sink" in self.calibration_stage:
                            pattern = "recent_sink"
                        raw_key_states,raw_value_states = self.eviction_tokens_head(raw_key_states,raw_value_states, 
                                                                           attn_weights, self.layer_idx,
                                                                           threshold, recent_tokens, self.calibration_stage,
                                                                           pattern = pattern)
                    else:
                        raw_key_states,raw_value_states = self.eviction_tokens(raw_key_states,raw_value_states, 
                                                                           attn_weights, self.layer_idx,
                                                                           threshold, recent_tokens, self.calibration_stage)                
                    past_key_value.key_cache[self.layer_idx] = raw_key_states
                    past_key_value.value_cache[self.layer_idx] = raw_value_states
                    # calibration
                    if "prefill_2_calibration" in self.calibration_stage:
                        key_states = raw_key_states
                        value_states = raw_value_states

                elif "prefill_1" in self.calibration_stage  and stage == 0:
                    if "head" in self.calibration_stage:
                        if "head_recent" in self.calibration_stage:
                            pattern = "recent"
                        elif "head_sink" in self.calibration_stage:
                            pattern = "sink"
                        elif "head_middle" in self.calibration_stage:
                            pattern = "middle"
                        elif "head_all" in self.calibration_stage:
                            pattern = "all"
                            
                        raw_key_states,raw_value_states = self.eviction_tokens_first_prefill_head(raw_key_states,raw_value_states, 
                                                                           attn_weights, self.layer_idx,
                                                                           threshold, recent_tokens, self.calibration_stage,
                                                                           pattern = pattern)
                    if "prefill_1_calibration" in self.calibration_stage:
                        key_states = raw_key_states
                        value_states = raw_value_states
                        
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            dropout_rate = self.attention_dropout if self.training else 0.0

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            if (
                self.config.use_sliding_window
                and getattr(self.config, "sliding_window", None) is not None
                and self.layer_idx >= self.config.max_window_layers
            ):
                sliding_window = self.config.sliding_window
            else:
                sliding_window = None

            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=dropout_rate,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                is_causal=self.is_causal,
            )
    
        if past_key_value is not None: 
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            if raw_len_past_key_value!=self.config.num_hidden_layers: # 第一次prefill
                self.recent_attn=self.get_attn(raw_query_states, raw_key_states)
                if self.kv_cache_eviction_prefill and not self.stage_eviction:
                    new_q_len = raw_key_states.shape[-2]
                    if "h2o" in self.calibration_stage:
                        self._update_kv_first_prefill_h2o(past_key_value,raw_key_states, raw_query_states,raw_value_states, new_q_len)
                    elif "streamingllm" in self.calibration_stage:
                        self._update_kv_first_prefill_streamingllm(past_key_value,raw_key_states, raw_query_states,raw_value_states, new_q_len)
                    elif "snapkv" in self.calibration_stage:
                        self._update_kv_first_prefill_snapkv(past_key_value,raw_key_states, raw_query_states,raw_value_states, new_q_len)
                    elif "pyramidkv" in self.calibration_stage:
                        self._update_kv_first_prefill_pyramidkv(past_key_value,raw_key_states, raw_query_states,raw_value_states, new_q_len)
                    elif "uncomp" in self.calibration_stage:
                        self._update_kv_first_prefill_uncomp(past_key_value,raw_key_states, raw_query_states,raw_value_states, new_q_len)
                    else:
                        self._update_kv_first_prefill(past_key_value,raw_key_states, raw_query_states,raw_value_states, new_q_len)
                else:
                    past_key_value.update(raw_key_states, raw_value_states, self.layer_idx, cache_kwargs)
            else:  # 第二次prefill和generate
                if self.stage_eviction:
                    if kv_seq_len != 1: # 第二次prefill 且驱逐
                        # 更新self.window
                        self.window_size = min(self.window_size,kv_seq_len)
                        self._update_kv_second_prefill(past_key_value,raw_key_states, raw_query_states,raw_value_states, past_key_value.key_cache[0].shape[2])
                    else:
                        if self.kv_cache_dynamic:
                            attn_weights = torch.matmul(raw_query_states, raw_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                            if attention_mask is not None:  # no matter the length, we just slice it
                                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                                attn_weights = attn_weights + causal_mask
                            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                            self._update_kv_generate(attn_weights,past_key_value,raw_key_states,raw_value_states,self.layer_idx)
                elif self.kv_cache_eviction_prefill:
                    if self.kv_cache_dynamic:
                        attn_weights = torch.matmul(raw_query_states, raw_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                        if attention_mask is not None:  # no matter the length, we just slice it
                            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                            attn_weights = attn_weights + causal_mask
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                        
                        if kv_seq_len != 1: # 第二次prefill 多个token处理
                            self._update_kv_input(attn_weights,kv_seq_len)
                        else:
                            self._update_kv_generate(attn_weights,past_key_value,raw_key_states,raw_value_states,self.layer_idx)
        
        if raw_len_past_key_value != self.config.num_hidden_layers and "uncomp_stage" in self.calibration_stage:
            attn_weights = torch.matmul(raw_query_states[..., -self.window_size:, :], raw_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]
            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(raw_query_states.dtype)
            manager.last_attn = attn_weights.sum(0).sum(0)[-8:].sum(0)
        
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
            
        return attn_output, attn_weights, past_key_value

class Qwen2ForCausalLMPCW(Qwen2ForCausalLM, ABC):
    _no_split_modules = ["Qwen2DecoderLayer"]
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: LlamaConfig, 
                 capacity = 512, n_windows=1, kv_cache_eviction=False,
                 kv_cache_dynamic=False,stage_eviction=False,
                 parallel_pattern=None,
                 calibration_mode=0,calibration_stage=None,
                 head_datas=None,
                 ):
        super(Qwen2ForCausalLM, self).__init__(config)
        # super().__init__(config)
        self.capacity =capacity
        self.manager = Manager()
        self.n_windows = n_windows
        self.parallel_pattern = parallel_pattern
        self.kv_cache_dynamic = kv_cache_dynamic
        # 确定config更改与否
        config.revise = False
        
        # raw code
        self.model = Qwen2ModelPCW(config, capacity = capacity, n_windows=n_windows, 
                                   kv_cache_eviction=kv_cache_eviction,
                                   kv_cache_dynamic=kv_cache_dynamic,
                                   stage_eviction=stage_eviction,
                                   calibration_mode=calibration_mode,
                                   calibration_stage=calibration_stage,
                                   manager=self.manager,
                                   head_datas=head_datas,
                                   )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma2 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if "ppl" in self.parallel_pattern and "default" not in self.parallel_pattern and labels is not None:
            ppl = torch.exp(loss)
            print("ppl is:{}".format(ppl))
            assert 1==0
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
      


class Qwen2ModelPCW(Qwen2Model, ABC):
    
    def __init__(self, config: LlamaConfig, 
              capacity = 512, n_windows=1, kv_cache_eviction=False,
                 kv_cache_dynamic=False,stage_eviction=False,
                 calibration_mode=0,calibration_stage=None,
                 head_datas=None,manager=None,
                 ):
        super(Qwen2Model, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerPCW(config, layer_idx, 
                                  capacity = capacity, n_windows=n_windows,
                                  kv_cache_eviction=kv_cache_eviction,
                                  head_datas=head_datas,
                                    kv_cache_dynamic=kv_cache_dynamic,
                                    stage_eviction=stage_eviction,
                                    calibration_mode=calibration_mode,
                                    manager=manager,
                                    calibration_stage=calibration_stage,
                                  ) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()
        
Qwen2_ATTENTION_CLASSES = {
    "eager": Qwen2FlashAttention2PCW,
    "flash_attention_2": Qwen2FlashAttention2PCW,
}

class Qwen2DecoderLayerPCW(Qwen2DecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None,
                 capacity = 512, n_windows=1, kv_cache_eviction=False,
                 kv_cache_dynamic=False,stage_eviction=False,
                 calibration_mode=0,calibration_stage=None,
                 head_datas=None, manager=None,
                 ):
        super(Qwen2DecoderLayer, self).__init__()
        # raw code 
        self.config = config
        self.manager = manager
        self.hidden_size = config.hidden_size
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen2_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx,
                                           capacity = capacity, n_windows=n_windows, 
                                           kv_cache_eviction=kv_cache_eviction,
                                           kv_cache_dynamic=kv_cache_dynamic,
                                           stage_eviction=stage_eviction,
                                           calibration_mode=calibration_mode,
                                           calibration_stage=calibration_stage,
                                           manager=manager,
                                           head_datas=head_datas,
                                           )
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        num_hidden_layers = self.config.num_hidden_layers
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        manager = self.manager
        if manager.last_attn!=None and self.self_attn.revise:
            indices = self.self_attn.select_indices
            residual = residual.gather(-2,indices.unsqueeze(-1).unsqueeze(0).expand(residual.size(0),-1,residual.size(-1)))
            if self.self_attn.layer_idx == (num_hidden_layers-1):
                manager.last_attn = None
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs