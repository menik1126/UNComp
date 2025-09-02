import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import (
    logging,
)
import os
from uncomp.uncomp_utils import init_uncomp
import math
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import time
import sys
from uncomp.utils.logger import Logger
from uncomp.utils.file_utils import (
    save_lists_to_csv,
) 
import uncomp.utils.metrics_utils as metrics_utils

from uncomp.utils.entropy_utils import (
    get_entropy_3_dimensions_single_batch,
    get_entropy_4_dimensions_single_batch,
    get_entropy_head_4_dimensions_single_batch,
    get_entropy_head_svd_4_dimensions_single_batch,
)

logger = Logger()

def LlamaDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    
    manager=None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    num_hidden_layers = manager.num_hidden_layers
    num_attention_heads = manager.num_attention_heads
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
    )
    if self.self_attn.kv_cluster.manager.last_attn!=None and self.self_attn.revise:
        indices = self.self_attn.select_indices
        residual = residual.gather(-2,indices.unsqueeze(-1).unsqueeze(0).expand(residual.size(0),-1,residual.size(-1)))
        if self.self_attn.layer_idx == (num_hidden_layers-1):
            manager.last_attn = None
    
    hidden_states = residual + hidden_states
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

def llama_attn_forward_Uncomp(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    manager=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    self.revise = False
    num_hidden_layers = manager.num_hidden_layers
    num_attention_heads = manager.num_attention_heads
    # hidden state 压缩前
    # get_memory_info = manager.func_utils_instance.get_memory_info
    # bytes_to_gb = manager.func_utils_instance.bytes_to_gb
    # torch.cuda.empty_cache()
    # before_reserved, before_allocated = get_memory_info()
    if manager.method_name in manager.hidden_delete_stage_and_ours and hidden_states.shape[-2] != 1:
        if "current" not in manager.method_name and "hidden" not in manager.method_name:
            if self.layer_idx != 0 :
                self.revise = True
                if num_hidden_layers == 40:
                    stage_label = [0,                           
                                   1,1,                         
                                   2,2,2,2,2, 2,2,2,2,2, 2,2,   
                                   3,3,3,3,3, 3,3,3,            
                                   4,4,4,4,4, 4,4,4,            
                                   5,5,5,5,                     
                                   6,6,                         
                                   7,7,7,                       
                                   ] 
                    attn_sum = manager.last_attn
                    min_len = 1536
                    allowance = (manager.max_token-min_len)//7
                    keep_seq_len = min_len+allowance*(7-stage_label[self.layer_idx])    
                else:
                    stage_label = [0,0, 
                                   1,1,1,1,1,1,1,1,1,1,1,1,1, 
                                   2,2,2,2,2,2,2,2,2,2,2, 
                                   3,3,3,3,3, 
                                   4, 
                                   ] 
                    attn_sum = manager.last_attn
                    min_len = 1536
                    allowance = (manager.max_token-min_len)//4
                    keep_seq_len = min_len+allowance*(4-stage_label[self.layer_idx])
                if "pyramidkv" in manager.method_name:
                    attn_sum = manager.last_attn
                    min_len = 512
                    max_len = manager.max_token-200
                    steps = (max_len-min_len)//(num_hidden_layers-1)
                    keep_seq_len = min_len+steps*(31-self.layer_idx)
                      
                if keep_seq_len > hidden_states.shape[-2]:
                    keep_seq_len = hidden_states.shape[-2]
                indices = attn_sum.topk(keep_seq_len,dim=-1).indices
                indices = indices.sort(dim=-1).values
                self.select_indices = indices
                raw_len = position_ids.size(-1)
                position_ids = torch.arange(0,indices.size(-1),device=indices.device).unsqueeze(0)
                position_ids = manager.last_position_ids.gather(-1,indices.unsqueeze(0))
                attention_mask = manager.last_attention_mask.gather(-2,indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,manager.last_attention_mask.shape[-1]))
                new_attention_mask = attention_mask.gather(-1,indices.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(-1,-1,attention_mask.shape[-2],-1))
                new_hidden_states = hidden_states.gather(-2,indices.unsqueeze(-1).unsqueeze(0).expand(hidden_states.size(0),-1,hidden_states.size(-1)))
                del hidden_states,attention_mask
                torch.cuda.empty_cache()
                hidden_states = new_hidden_states
                attention_mask = new_attention_mask
                
            manager.last_position_ids = position_ids
            manager.last_attention_mask = attention_mask
        elif "hidden_1" in manager.method_name:
            if self.layer_idx != 0 :
                self.revise = True
                bsz, q_len, _ = hidden_states.size()
                
                # 本层hidden_states之间相似度度量
                similarity = torch.matmul(hidden_states[:,-8:,:], hidden_states.transpose(1, 2)).sum(0).sum(0)
                last_8_hidden = hidden_states[:, -8:, :]
                hidden_states_norm = F.normalize(hidden_states, dim=2)
                last_8_hidden_norm = F.normalize(last_8_hidden, dim=2)
                cosine_similarity = torch.matmul(last_8_hidden_norm, hidden_states_norm.transpose(1, 2)).sum(0).sum(0)
                manager.last_attn = cosine_similarity
                
                
                if num_hidden_layers == 40:
                    stage_label = [0,                           
                                   1,1,                         
                                   2,2,2,2,2, 2,2,2,2,2, 2,2,   
                                   3,3,3,3,3, 3,3,3,            
                                   4,4,4,4,4, 4,4,4,            
                                   5,5,5,5,                     
                                   6,6,                         
                                   7,7,7,                       
                                   ] 
                    attn_sum = manager.last_attn
                    min_len = 1536
                    allowance = (manager.max_token-min_len)//7
                    keep_seq_len = min_len+allowance*(7-stage_label[self.layer_idx])    
                else:
                    stage_label = [0,0, 
                                   1,1,1,1,1,1,1,1,1,1,1,1,1, 
                                   2,2,2,2,2,2,2,2,2,2,2, 
                                   3,3,3,3,3, 
                                   4, 
                                   ] 
                    attn_sum = manager.last_attn
                    min_len = 1536
                    allowance = (manager.max_token-min_len)//4
                    keep_seq_len = min_len+allowance*(4-stage_label[self.layer_idx])
                if "pyramidkv" in manager.method_name:
                    min_len = 512
                    max_len = manager.max_token-200
                    steps = (max_len-min_len)//(num_hidden_layers-1)
                    keep_seq_len = min_len+steps*(31-self.layer_idx)
                      
                if keep_seq_len > hidden_states.shape[-2]:
                    keep_seq_len = hidden_states.shape[-2]
                indices = attn_sum.topk(keep_seq_len,dim=-1).indices
                indices = indices.sort(dim=-1).values
                self.select_indices = indices
                raw_len = position_ids.size(-1)
                position_ids = torch.arange(0,indices.size(-1),device=indices.device).unsqueeze(0)
                position_ids = manager.last_position_ids.gather(-1,indices.unsqueeze(0))
                attention_mask = manager.last_attention_mask.gather(-2,indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,manager.last_attention_mask.shape[-1]))
                attention_mask = attention_mask.gather(-1,indices.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(-1,-1,attention_mask.shape[-2],-1))
                hidden_states = hidden_states.gather(-2,indices.unsqueeze(-1).unsqueeze(0).expand(hidden_states.size(0),-1,hidden_states.size(-1)))

            manager.last_position_ids = position_ids
            manager.last_attention_mask = attention_mask
        elif "hidden_2" in manager.method_name:
            if self.layer_idx != 0 :
                self.revise = True
                bsz, q_len, _ = hidden_states.size()
                query_states_temp = self.q_proj(hidden_states)
                key_states_temp = self.k_proj(hidden_states)
                value_states_temp = self.v_proj(hidden_states)

                query_states_temp = query_states_temp.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states_temp = key_states_temp.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states_temp = value_states_temp.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                position_ids = manager.last_position_ids
                attention_mask = manager.last_attention_mask
                cos, sin = self.rotary_emb(value_states_temp, position_ids)
                query_states_temp, key_states_temp = apply_rotary_pos_emb(query_states_temp, key_states_temp, cos, sin)
                
                key_states_temp = repeat_kv(key_states_temp, self.num_key_value_groups)
                value_states_temp = repeat_kv(value_states_temp, self.num_key_value_groups)

                attn_weights_temp = torch.matmul(query_states_temp, key_states_temp.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, :, : key_states_temp.shape[-2]]
                    attn_weights_temp = attn_weights_temp + causal_mask

                # upcast attention to fp32
                attn_weights_temp = nn.functional.softmax(attn_weights_temp, dim=-1, dtype=torch.float32).to(query_states_temp.dtype)
                attn_weights_temp = nn.functional.dropout(attn_weights_temp, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights_temp, value_states_temp)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                if self.config.pretraining_tp > 1:
                    attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                    o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                    attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
                else:
                    attn_output = self.o_proj(attn_output)
                # 本层hidden_states之间相似度度量
                last_8_hidden = attn_output[:, -8:, :]
                hidden_states_norm = F.normalize(attn_output, dim=2)
                last_8_hidden_norm = F.normalize(last_8_hidden, dim=2)
                cosine_similarity = torch.matmul(last_8_hidden_norm, hidden_states_norm.transpose(1, 2)).sum(0).sum(0)
                manager.last_attn = cosine_similarity
                
                
                if num_hidden_layers == 40:
                    stage_label = [0,                           
                                   1,1,                         
                                   2,2,2,2,2, 2,2,2,2,2, 2,2,   
                                   3,3,3,3,3, 3,3,3,            
                                   4,4,4,4,4, 4,4,4,            
                                   5,5,5,5,                     
                                   6,6,                         
                                   7,7,7,                       
                                   ] 
                    attn_sum = manager.last_attn
                    min_len = 1536
                    allowance = (manager.max_token-min_len)//7
                    keep_seq_len = min_len+allowance*(7-stage_label[self.layer_idx])    
                else:
                    stage_label = [0,0, 
                                   1,1,1,1,1,1,1,1,1,1,1,1,1, 
                                   2,2,2,2,2,2,2,2,2,2,2, 
                                   3,3,3,3,3, 
                                   4, 
                                   ] 
                    attn_sum = manager.last_attn
                    min_len = 1536
                    allowance = (manager.max_token-min_len)//4
                    keep_seq_len = min_len+allowance*(4-stage_label[self.layer_idx])
                if "pyramidkv" in manager.method_name:
                    min_len = 512
                    max_len = manager.max_token-200
                    steps = (max_len-min_len)//(num_hidden_layers-1)
                    keep_seq_len = min_len+steps*(31-self.layer_idx)
                      
                if keep_seq_len > hidden_states.shape[-2]:
                    keep_seq_len = hidden_states.shape[-2]
                indices = attn_sum.topk(keep_seq_len,dim=-1).indices
                indices = indices.sort(dim=-1).values
                self.select_indices = indices
                raw_len = position_ids.size(-1)
                position_ids = torch.arange(0,indices.size(-1),device=indices.device).unsqueeze(0)
                position_ids = manager.last_position_ids.gather(-1,indices.unsqueeze(0))
                attention_mask = manager.last_attention_mask.gather(-2,indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,manager.last_attention_mask.shape[-1]))
                attention_mask = attention_mask.gather(-1,indices.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(-1,-1,attention_mask.shape[-2],-1))
                hidden_states = hidden_states.gather(-2,indices.unsqueeze(-1).unsqueeze(0).expand(hidden_states.size(0),-1,hidden_states.size(-1)))

            manager.last_position_ids = position_ids
            manager.last_attention_mask = attention_mask
        elif "hidden_3" in manager.method_name:
            if self.layer_idx != 0 :
                self.revise = True
                bsz, q_len, _ = hidden_states.size()
                query_states_temp = self.q_proj(hidden_states)
                key_states_temp = self.k_proj(hidden_states)
                value_states_temp = self.v_proj(hidden_states)

                query_states_temp = query_states_temp.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states_temp = key_states_temp.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states_temp = value_states_temp.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                position_ids = manager.last_position_ids
                attention_mask = manager.last_attention_mask
                cos, sin = self.rotary_emb(value_states_temp, position_ids)
                query_states_temp, key_states_temp = apply_rotary_pos_emb(query_states_temp, key_states_temp, cos, sin)
                
                key_states_temp = repeat_kv(key_states_temp, self.num_key_value_groups)
                value_states_temp = repeat_kv(value_states_temp, self.num_key_value_groups)

                attn_weights_temp = torch.matmul(query_states_temp, key_states_temp.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, :, : key_states_temp.shape[-2]]
                    attn_weights_temp = attn_weights_temp + causal_mask

                # upcast attention to fp32
                attn_weights_temp = nn.functional.softmax(attn_weights_temp, dim=-1, dtype=torch.float32).to(query_states_temp.dtype)
                attn_weights_temp = nn.functional.dropout(attn_weights_temp, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights_temp, value_states_temp)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                if self.config.pretraining_tp > 1:
                    attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                    o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                    attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
                else:
                    attn_output = self.o_proj(attn_output)

                
                def gaussian_kernel_diff_dim(A, B, sigma=1.0):
                    """
                    针对形状不完全相同的三维张量计算成对高斯核相似度。
                    Args:
                        A: Tensor，形状为 [batch_size, seq_len1, dim]
                        B: Tensor，形状为 [batch_size, seq_len2, dim]
                        sigma: float，高斯核的宽度参数
                    Returns:
                        相似度张量，形状为 [batch_size, seq_len1, seq_len2]
                    """
                    # 检查最后一个维度是否相同
                    assert A.size(-1) == B.size(-1), "The last dimension of A and B must match."
                    
                    # 计算差值
                    # 使用广播机制将 A 和 B 扩展为 [batch_size, seq_len1, seq_len2, dim]
                    diff = A.unsqueeze(2) - B.unsqueeze(1)  # [batch_size, seq_len1, seq_len2, dim]
                    print("diff: ", diff)
                    # 欧几里得距离的平方
                    dist_squared = torch.sum(diff ** 2, dim=-1)  # 对最后一个维度求和，形状为 [batch_size, seq_len1, seq_len2]
                    
                    # 应用高斯核公式
                    similarity_matrix = torch.exp(-dist_squared / (2 * sigma ** 2))  # [batch_size, seq_len1, seq_len2]
                    return similarity_matrix
                
                
                last_n_hidden = hidden_states[:, -64:, :]
                last_n_output = attn_output[:, -64:, :]
                if "hidden_3_1" in manager.method_name:
                    similarity = gaussian_kernel_diff_dim(last_n_hidden, hidden_states, sigma=1.0).sum(0).sum(0)
                elif "hidden_3_2" in manager.method_name:
                    similarity = gaussian_kernel_diff_dim(last_n_output, attn_output, sigma=1.0).sum(0).sum(0)
                elif "hidden_3_3" in manager.method_name:
                    similarity = gaussian_kernel_diff_dim(last_n_output, hidden_states, sigma=1.0).sum(0).sum(0)
                elif "hidden_3_4" in manager.method_name:
                    similarity = gaussian_kernel_diff_dim(last_n_hidden, attn_output, sigma=1.0).sum(0).sum(0)
                
                else:
                    assert False, "method_name error"
                manager.last_attn = similarity
                
                similarity1 = gaussian_kernel_diff_dim(last_n_output, hidden_states, sigma=1.0).sum(0).sum(0)
                similarity2 = gaussian_kernel_diff_dim(last_n_hidden, attn_output, sigma=1.0).sum(0).sum(0)
                difference = torch.abs(similarity1 - similarity2)
                similarity1_max = difference.max()
                similarity1_min = difference.min()
                print("difference: ", difference)
                print("similarity1_max: ", similarity1_max)
                
                assert 1==0
                
                if num_hidden_layers == 40:
                    stage_label = [0,                           
                                   1,1,                         
                                   2,2,2,2,2, 2,2,2,2,2, 2,2,   
                                   3,3,3,3,3, 3,3,3,            
                                   4,4,4,4,4, 4,4,4,            
                                   5,5,5,5,                     
                                   6,6,                         
                                   7,7,7,                       
                                   ] 
                    attn_sum = manager.last_attn
                    min_len = 1536
                    allowance = (manager.max_token-min_len)//7
                    keep_seq_len = min_len+allowance*(7-stage_label[self.layer_idx])    
                else:
                    stage_label = [0,0, 
                                   1,1,1,1,1,1,1,1,1,1,1,1,1, 
                                   2,2,2,2,2,2,2,2,2,2,2, 
                                   3,3,3,3,3, 
                                   4, 
                                   ] 
                    attn_sum = manager.last_attn
                    min_len = 1536
                    allowance = (manager.max_token-min_len)//4
                    keep_seq_len = min_len+allowance*(4-stage_label[self.layer_idx])
                if "pyramidkv" in manager.method_name:
                    min_len = 512
                    max_len = manager.max_token-200
                    steps = (max_len-min_len)//(num_hidden_layers-1)
                    keep_seq_len = min_len+steps*(31-self.layer_idx)
                      
                if keep_seq_len > hidden_states.shape[-2]:
                    keep_seq_len = hidden_states.shape[-2]
                indices = attn_sum.topk(keep_seq_len,dim=-1).indices
                indices = indices.sort(dim=-1).values
                self.select_indices = indices
                raw_len = position_ids.size(-1)
                position_ids = torch.arange(0,indices.size(-1),device=indices.device).unsqueeze(0)
                position_ids = manager.last_position_ids.gather(-1,indices.unsqueeze(0))
                attention_mask = manager.last_attention_mask.gather(-2,indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,manager.last_attention_mask.shape[-1]))
                attention_mask = attention_mask.gather(-1,indices.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(-1,-1,attention_mask.shape[-2],-1))
                hidden_states = hidden_states.gather(-2,indices.unsqueeze(-1).unsqueeze(0).expand(hidden_states.size(0),-1,hidden_states.size(-1)))

            manager.last_position_ids = position_ids
            manager.last_attention_mask = attention_mask       
        else:
            if self.layer_idx != 0 :
                self.revise = True
                # 提前计算
                bsz, q_len, _ = hidden_states.size()
                query_states_temp = self.q_proj(hidden_states)
                key_states_temp = self.k_proj(hidden_states)
                value_states_temp = self.v_proj(hidden_states)

                query_states_temp = query_states_temp.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states_temp = key_states_temp.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states_temp = value_states_temp.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                position_ids = manager.last_position_ids
                attention_mask = manager.last_attention_mask
                cos, sin = self.rotary_emb(value_states_temp, position_ids)
                query_states_temp, key_states_temp = apply_rotary_pos_emb(query_states_temp, key_states_temp, cos, sin)
                
                key_states_temp = repeat_kv(key_states_temp, self.num_key_value_groups)
                key_states_temp = repeat_kv(key_states_temp, self.num_key_value_groups)

                attn_weights_temp = torch.matmul(query_states_temp, key_states_temp.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, :, : key_states_temp.shape[-2]]
                    attn_weights_temp = attn_weights_temp + causal_mask

                # upcast attention to fp32
                attn_weights_temp = nn.functional.softmax(attn_weights_temp, dim=-1, dtype=torch.float32).to(query_states_temp.dtype)
                attn_weights_temp = nn.functional.dropout(attn_weights_temp, p=self.attention_dropout, training=self.training)
                manager.last_attn = attn_weights_temp.sum(0).sum(0)[-8:].sum(0)
                
                if num_hidden_layers == 40:
                    stage_label = [0,                           
                                   1,1,                         
                                   2,2,2,2,2, 2,2,2,2,2, 2,2,   
                                   3,3,3,3,3, 3,3,3,            
                                   4,4,4,4,4, 4,4,4,            
                                   5,5,5,5,                     
                                   6,6,                         
                                   7,7,7,                       
                                   ] 
                    attn_sum = manager.last_attn
                    min_len = 1536
                    allowance = (manager.max_token-min_len)//7
                    keep_seq_len = min_len+allowance*(7-stage_label[self.layer_idx])    
                else:
                    stage_label = [0,0, 
                                   1,1,1,1,1,1,1,1,1,1,1,1,1, 
                                   2,2,2,2,2,2,2,2,2,2,2, 
                                   3,3,3,3,3, 
                                   4, 
                                   ] 
                    attn_sum = manager.last_attn
                    min_len = 1536
                    allowance = (manager.max_token-min_len)//4
                    keep_seq_len = min_len+allowance*(4-stage_label[self.layer_idx])
                if "pyramidkv" in manager.method_name:
                    min_len = 512
                    max_len = manager.max_token-200
                    steps = (max_len-min_len)//(num_hidden_layers-1)
                    keep_seq_len = min_len+steps*(31-self.layer_idx)
                      
                if keep_seq_len > hidden_states.shape[-2]:
                    keep_seq_len = hidden_states.shape[-2]
                indices = attn_sum.topk(keep_seq_len,dim=-1).indices
                indices = indices.sort(dim=-1).values
                self.select_indices = indices
                raw_len = position_ids.size(-1)
                position_ids = torch.arange(0,indices.size(-1),device=indices.device).unsqueeze(0)
                position_ids = manager.last_position_ids.gather(-1,indices.unsqueeze(0))
                attention_mask = manager.last_attention_mask.gather(-2,indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,manager.last_attention_mask.shape[-1]))
                attention_mask = attention_mask.gather(-1,indices.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(-1,-1,attention_mask.shape[-2],-1))
                hidden_states = hidden_states.gather(-2,indices.unsqueeze(-1).unsqueeze(0).expand(hidden_states.size(0),-1,hidden_states.size(-1)))

            manager.last_position_ids = position_ids
            manager.last_attention_mask = attention_mask
    # torch.cuda.empty_cache()
    # after_reserved, after_allocated = get_memory_info()
    # print(f"\nlayer {self.layer_idx} Memory reserved before: {bytes_to_gb(before_reserved)} GB, after: {bytes_to_gb(after_reserved)} GB")
    # print(f"layer {self.layer_idx} Memory allocated before: {bytes_to_gb(before_allocated)} GB, after: {bytes_to_gb(after_allocated)} GB")
    
    # hidden states
    if manager.method_name in manager.rope:
        if manager.method_name in manager.rope_survey:
            # manager.rope_survey_dict["hidden_states"].append(get_entropy_3_dimensions_single_batch(hidden_states))
            logger.info(f"in layer: {self.layer_idx}")
    
    bsz, q_len, _ = hidden_states.size()
    manager.q_len = q_len
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len 
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            
    # rope 前
    if manager.method_name in manager.rope_position_ids_control:
        if self.layer_idx >= num_hidden_layers - manager.rope_position_ids_control_dict["layer"]:
            if "elongate" in manager.method_name:
                position_ids = torch.round(position_ids*manager.rope_position_ids_control_dict["elongate_ratio"]).long()
                position_ids = torch.clamp(position_ids, max=4092)
            elif "narrow" in manager.method_name:
                if "dynamic" in manager.method_name:
                    position_ids = torch.round(position_ids*manager.rope_position_ids_control_dict[self.layer_idx]).long()
                else:
                    position_ids = torch.round(position_ids*manager.rope_position_ids_control_dict["narrow_ratio"]).long()
            
    cos, sin = self.rotary_emb(value_states, position_ids)
    
    if manager.method_name in manager.rope:
        clip = False
        if manager.method_name in manager.rope_survey:
            manager.rope_survey_dict["query_states"].append(get_entropy_4_dimensions_single_batch(query_states))
            manager.rope_survey_dict["key_states"].append(get_entropy_4_dimensions_single_batch(key_states))
            manager.rope_survey_dict["value_states"].append(get_entropy_4_dimensions_single_batch(value_states))
            
            manager.rope_survey_head_dict["query_states_head"].append(get_entropy_head_4_dimensions_single_batch(query_states))
            manager.rope_survey_head_dict["key_states_head"].append(get_entropy_head_4_dimensions_single_batch(key_states))
            manager.rope_survey_head_dict["value_states_head"].append(get_entropy_head_4_dimensions_single_batch(value_states))
        elif manager.method_name in manager.rope_clip_layer_single_layer:
            clip_layer_idx = manager.rope_clip_layer_single_layer_dict["clip_layer"]
            clip = True
        elif manager.method_name in manager.rope_clip_layer_multi_layer:
            clip_layer_idx = manager.rope_clip_layer_multi_layer_dict["clip_layers"]
            clip = True
        elif manager.method_name in manager.rope_correlation:
            # logger.info(f"manager.rope_correlation_dict[\"query_before_rope\"] : {manager.rope_correlation_dict['query_before_rope']}")
            manager.rope_correlation_dict["query_before_rope"]=get_entropy_head_4_dimensions_single_batch(query_states)
            manager.rope_correlation_dict["key_before_rope"]=get_entropy_head_4_dimensions_single_batch(key_states)
        
        if clip == True:
            if self.layer_idx not in clip_layer_idx: # 不是减去的层
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            else:
                logger.debug(f"clip layer: {self.layer_idx}")
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    # rope 后
    if manager.method_name in manager.rope:
        if manager.method_name in manager.rope_survey:
            manager.rope_survey_dict["query_states_rope"].append(get_entropy_4_dimensions_single_batch(query_states))
            manager.rope_survey_dict["key_states_rope"].append(get_entropy_4_dimensions_single_batch(key_states))
            
            manager.rope_survey_head_dict["query_states_head_rope"].append(get_entropy_head_4_dimensions_single_batch(query_states))
            manager.rope_survey_head_dict["key_states_head_rope"].append(get_entropy_head_4_dimensions_single_batch(key_states))
            
            manager.rope_survey_head_dict["query_states_svd32_head_rope"].append(get_entropy_head_svd_4_dimensions_single_batch(query_states,32))
            manager.rope_survey_head_dict["key_states_svd32_head_rope"].append(get_entropy_head_svd_4_dimensions_single_batch(key_states,32))
            
            manager.rope_survey_head_dict["query_states_svd16_head_rope"].append(get_entropy_head_svd_4_dimensions_single_batch(query_states,16))
            manager.rope_survey_head_dict["key_states_svd16_head_rope"].append(get_entropy_head_svd_4_dimensions_single_batch(key_states,16))
            
            manager.rope_survey_head_dict["query_states_svd64_head_rope"].append(get_entropy_head_svd_4_dimensions_single_batch(query_states,64))
            manager.rope_survey_head_dict["key_states_svd64_head_rope"].append(get_entropy_head_svd_4_dimensions_single_batch(key_states,64))
            
            manager.rope_survey_head_dict["query_states_svd96_head_rope"].append(get_entropy_head_svd_4_dimensions_single_batch(query_states,96))
            manager.rope_survey_head_dict["key_states_svd96_head_rope"].append(get_entropy_head_svd_4_dimensions_single_batch(key_states,96))
            
            if self.layer_idx == num_hidden_layers - 1 :
                keys_to_save = ["query_states_head_rope","key_states_head_rope","value_states_head","query_states_head","key_states_head"]
                keys_to_save = ["query_states_svd32_head_rope","key_states_svd32_head_rope"]
                keys_to_save = ["query_states_svd16_head_rope","key_states_svd16_head_rope"]
                keys_to_save = ["query_states_svd64_head_rope","key_states_svd64_head_rope"]
                keys_to_save = ["query_states_svd96_head_rope","key_states_svd96_head_rope"]
                
                
                # 保存32*32
                save_lists_to_csv(manager.rope_survey_head_dict,keys_to_save,manager.rope_survey_head_dict["save_path"])
                
                
                print("\nhidden_states:",manager.rope_survey_dict["hidden_states"])
                print("\nhidden_states*num_hidden_layers:",list(map(lambda x: x * num_hidden_layers, manager.rope_survey_dict["hidden_states"])))
                print("\nquery_states:",manager.rope_survey_dict["query_states"])
                print("key_states",manager.rope_survey_dict["key_states"])
                print("value_states",manager.rope_survey_dict["value_states"])
                print("\nquery_states_rope",manager.rope_survey_dict["query_states_rope"])
                print("key_states_rope",manager.rope_survey_dict["key_states_rope"])
                sys.exit(0)
        elif manager.method_name in manager.rope_correlation:
            manager.rope_correlation_dict["query_after_rope"]=get_entropy_head_4_dimensions_single_batch(query_states)
            manager.rope_correlation_dict["key_after_rope"]=get_entropy_head_4_dimensions_single_batch(key_states)
            
            correlation_q,_ = metrics_utils.z_score_pearsonr(manager.rope_correlation_dict["query_before_rope"],manager.rope_correlation_dict["query_after_rope"])
            correlation_k,_ = metrics_utils.z_score_pearsonr(manager.rope_correlation_dict["key_before_rope"],manager.rope_correlation_dict["key_after_rope"])
            print(f"{manager.rope_correlation_dict['dataname']},{self.layer_idx},{correlation_q},{correlation_k}") 
            
            current_path = os.getcwd()+"/save_files/correlation_z_score/"
            logger.debug(f"file path: {current_path} ")
            filename = f"{manager.rope_correlation_dict['dataname']}.csv"
            full_path = os.path.join(current_path, filename)
            with open(full_path, 'a', newline='') as file:
                # 只写入最后两列：correlation_q 和 correlation_k
                file.write(f"{correlation_q},{correlation_k}\n")
            
            # # 清空所有列表，并确保只清空值为列表的项
            # for key, value in manager.rope_correlation_dict.items():
            #     if isinstance(value, list):
            #         value.clear()

    if past_key_value is not None:
        # logger.info(f"position_ids is: {position_ids}")
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        raw_key_states_shape = key_states.shape[-2]
        if key_states.shape[-2] == kv_seq_len:
            init_uncomp(self, num_hidden_layers=self.config.num_hidden_layers,manager=manager)
            self.kv_seq_len = kv_seq_len
        elif key_states.shape[-2] != kv_seq_len:
            self.kv_seq_len += q_len    
            if manager.method_name in manager.delet_head_set:
                key_states, value_states = self.kv_cluster.update_past_key_value(past_key_value, key_states, value_states, self.layer_idx, 1)
            elif manager.method_name in manager.head_granularity:
                key_states, value_states = self.kv_cluster.update_past_key_value(past_key_value, key_states, value_states, self.layer_idx, 0)
            elif manager.method_name in manager.last_process: #最后一层处理
                key_states, value_states = self.kv_cluster.update_all_past_key_values(1, past_key_value, key_states, value_states, self.layer_idx, None)
            else:          
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if manager.method_name in manager.head_granularity and raw_key_states_shape != kv_seq_len :
        if not isinstance(key_states, list) and manager.method_name not in manager.delet_head_set:
            raise ValueError("key_states is not a list")
        attn_weights,attn_outputs = [], []
        if manager.method_name in manager.delet_head_set:
            max_similarity_indices = self.kv_cluster.similarity
            attn_weights = torch.matmul(query_states[:,self.kv_cluster.head_pattern,:,:], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:  
                causal_mask = attention_mask[:,:, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
                
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.zeros(bsz, self.num_heads, q_len, self.head_dim, dtype=query_states.dtype, device=query_states.device)
            attn_output_new = torch.matmul(attn_weights, value_states)
            attn_output[:,self.kv_cluster.head_pattern,:,:] = attn_output_new
            attn_output[:,self.kv_cluster.head_indices1[:num_attention_heads-manager.select_topk],:,:] = attn_output_new[:,max_similarity_indices,:,:]

            if past_key_value is not None:
                self.kv_cluster.update_head_kv_generate(past_key_value, attn_weights, key_states, value_states, self.layer_idx, cache_kwargs)         
        else:
            """stream method
            assert len(key_states) == manager.num_kv_groups, \
                f"Expected {manager.num_kv_groups} KV groups, but got {len(key_states)}"
            batch_size, total_heads, seq_len, head_dim = query_states.size()
            attn_output = torch.empty(
                (batch_size, total_heads, seq_len, head_dim),
                dtype=query_states.dtype,
                device=query_states.device
            )
            scale = 1.0 / math.sqrt(self.head_dim)
            for i, (key_state, value_state, stream) in enumerate(zip(key_states, value_states, manager.streams)):
                if len(self.kv_cluster.head_pattern[i]) == 0:
                    continue
                stream.wait_stream(torch.cuda.current_stream()) # 等待变量全部储备完成
                with torch.cuda.stream(stream):
                    # 计算复杂度: 两个 1*16*512*128 相乘
                    query_state = query_states[:, self.kv_cluster.head_pattern[i], :, :]
                    attn_weight = torch.matmul(query_state, key_state.transpose(2, 3)) * scale
                    if attention_mask is not None:
                        causal_mask = attention_mask[:, :, :, :key_state.shape[-2]]
                        attn_weight = attn_weight + causal_mask
                    
                    attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    attn_weight = nn.functional.dropout(
                        attn_weight, 
                        p=self.attention_dropout, 
                        training=self.training
                    )
                    attn_output_item = torch.matmul(attn_weight, value_state)
                    attn_output[:, self.kv_cluster.head_pattern[i], :, :] = attn_output_item
                    # outputs.append([i,attn_output_item])
                    if past_key_value is not None:
                        if manager.method_name not in manager.not_update:
                            self.kv_cluster.update_head_kv_generate(past_key_value, attn_weight, key_state, value_state, self.layer_idx,cache_kwargs, single=i)   
            for stream in manager.streams:
                torch.cuda.current_stream().wait_stream(stream) 
            """
            """for removed
            query_state = query_states[:,self.kv_cluster.head_pattern[0],:,:]
            attn_weight = torch.matmul(query_state, key_states[0].transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:   
                causal_mask = attention_mask[:, :, :, : key_states[0].shape[-2]]
                attn_weight = attn_weight + causal_mask
            attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights.append(attn_weight)             
            attn_weight = nn.functional.dropout(attn_weight, p=self.attention_dropout, training=self.training)
            attn_output_item = torch.matmul(attn_weight, value_states[0])
            attn_outputs.append(attn_output_item)
            
            query_state = query_states[:,self.kv_cluster.head_pattern[1],:,:]
            attn_weight = torch.matmul(query_state, key_states[1].transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:   
                causal_mask = attention_mask[:, :, :, : key_states[1].shape[-2]]
                attn_weight = attn_weight + causal_mask
            attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights.append(attn_weight)             
            attn_weight = nn.functional.dropout(attn_weight, p=self.attention_dropout, training=self.training)
            attn_output_item = torch.matmul(attn_weight, value_states[1])
            attn_outputs.append(attn_output_item)
            
            attn_output = torch.cat(attn_outputs,dim=1)
            for i in range(len(key_states)):
                if len(self.kv_cluster.head_pattern[i]) != 0:
                    attn_output[:,self.kv_cluster.head_pattern[i],:,:] = attn_outputs[i]
            if past_key_value is not None:
                if manager.method_name not in manager.not_update:
                    self.kv_cluster.update_head_kv_generate(past_key_value, attn_weights, key_states, value_states, self.layer_idx, cache_kwargs)
            """
            for i, (key_state,value_state) in enumerate(zip(key_states,value_states)):
                query_state = query_states[:,self.kv_cluster.head_pattern[i],:,:]
                attn_weight = torch.matmul(query_state, key_state.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:   
                    causal_mask = attention_mask[:, :, :, : key_state.shape[-2]]
                    attn_weight = attn_weight + causal_mask
                attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)      
                attn_weight = nn.functional.dropout(attn_weight, p=self.attention_dropout, training=self.training)
                attn_output_item = torch.matmul(attn_weight, value_state)
                attn_outputs.append(attn_output_item)
                if past_key_value is not None:
                    if manager.method_name not in manager.not_update:
                        self.kv_cluster.update_head_kv_generate(past_key_value, attn_weight, key_state, value_state, self.layer_idx, cache_kwargs, single=i)
            attn_output = torch.cat(attn_outputs,dim=1)
            for i in range(len(key_states)):
                if len(self.kv_cluster.head_pattern[i]) != 0:
                    attn_output[:,self.kv_cluster.head_pattern[i],:,:] = attn_outputs[i]
    elif manager.method_name in manager.chai and self.layer_idx >= 0:
        cluster_assignment_log_per_example = dict()
        if raw_key_states_shape == kv_seq_len:
            num_examples, num_org_heads, seq_len, head_dim = query_states.shape
            query_states_four = query_states[:, :, :5, :]
            key_states_four = key_states[:, :, :5, :]
            scores_four = F.softmax((torch.matmul(query_states_four, key_states_four.transpose(2, 3))/ math.sqrt(self.head_dim)).float(),dim=-1)
            scores_four_numpy = scores_four.cpu().numpy()
            scores_new_xk_xq = torch.zeros(
                [num_examples, num_org_heads, seq_len, seq_len],
                device=query_states.device,
                dtype=query_states.dtype,
            )
            xk_new = torch.zeros(
                [num_examples, manager.chai_layers[self.layer_idx], seq_len, head_dim],
                dtype=key_states.dtype,
                device=key_states.device,
            )
            xq_new = torch.zeros(
                [num_examples, manager.chai_layers[self.layer_idx], seq_len, head_dim],
                dtype=query_states.dtype,
                device=query_states.device,
            )
            for ex_id in range(num_examples):
                assert num_examples == 1
                temp_data = dict()
                ex_id_score = scores_four_numpy[ex_id, :]
                sequence_length_example = ex_id_score.shape[1]
                
                num_heads = ex_id_score.shape[0]
                first_sample_score = ex_id_score.reshape((num_heads, -1))
                dist_arr = cdist(
                    first_sample_score, first_sample_score, metric="cosine"
                )
                cluster = AgglomerativeClustering(
                        n_clusters=manager.chai_layers[self.layer_idx],
                        metric="precomputed",
                       linkage="average",
                )
                cluster = cluster.fit(dist_arr)
                
                cluster_assignment = cluster.labels_
                self.grouping = cluster_assignment
                for cluster_idx in range(manager.chai_layers[self.layer_idx]):
                    grouped_heads = np.where(cluster_assignment == cluster_idx)[
                        0
                    ].tolist()
                    xk_new[ex_id, cluster_idx, :, :] = key_states[ex_id, grouped_heads[0], :, :]
                    xq_new[ex_id, cluster_idx, :, :] = query_states[ex_id, grouped_heads[0], :, :]
                    temp_data[cluster_idx] = grouped_heads
                cluster_assignment_log_per_example[ex_id] = temp_data
        else:
            num_examples, num_org_heads, seq_len, head_dim = key_states.shape
            scores_new_xk_xq = torch.zeros(
                [num_examples, num_org_heads, 1, seq_len],
                device=query_states.device,
                dtype=query_states.dtype,
            )
            xk_new = torch.zeros(
                [num_examples, manager.chai_layers[self.layer_idx], seq_len, head_dim],
                dtype=key_states.dtype,
                device=key_states.device,
            )
            xq_new = torch.zeros(
                [num_examples, manager.chai_layers[self.layer_idx], 1, head_dim],
                dtype=query_states.dtype,
                device=query_states.device,
            )
            cluster_assignment = self.grouping
            for ex_id in range(num_examples):
                temp_data = dict()
                for cluster_idx in range(manager.chai_layers[self.layer_idx]):
                    grouped_heads = np.where(cluster_assignment == cluster_idx)[
                        0
                    ].tolist()
                    xk_new[ex_id, cluster_idx, :, :] = key_states[
                        ex_id, grouped_heads[0], :, :
                    ]
                    xq_new[ex_id, cluster_idx, :, :] = query_states[
                        ex_id, grouped_heads[0], :, :
                    ]
                    temp_data[cluster_idx] = grouped_heads
                cluster_assignment_log_per_example[ex_id] = temp_data

        scores_new_temp = torch.matmul(xq_new, xk_new.transpose(2, 3)) / math.sqrt(self.head_dim)
        for ex_id in range(num_examples):
            for cluster_idx in range(manager.chai_layers[self.layer_idx]):
                scores_new_xk_xq[
                    ex_id,
                    cluster_assignment_log_per_example[ex_id][cluster_idx],
                    :,
                    :,
                ] = scores_new_temp[ex_id, cluster_idx, :, :]
        
        if attention_mask is not None:  
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            scores_new_xk_xq = scores_new_xk_xq + attention_mask
            
        scores_new_xk_xq = F.softmax(scores_new_xk_xq.float(), dim=-1).type_as(query_states)
        attn_weights = scores_new_xk_xq
        
        attn_output = torch.matmul(attn_weights, value_states)  

        if raw_key_states_shape == kv_seq_len:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attn_weights, attn_weights)
            if manager.method_name in manager.head_granularity:
                self.kv_cluster.update_past_key_value(past_key_value, key_states_compress, value_states_compress, self.layer_idx, 0)
            elif manager.method_name in  manager.last_process or manager.method_name in manager.search:
                if self.layer_idx == (num_hidden_layers-1):
                    self.kv_cluster.update_all_past_key_values(0, past_key_value, key_states, value_states, None, None)
            else:
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)               
    else:
        if not isinstance(key_states, torch.Tensor):
            print("type(key_states):",type(key_states))
            raise ValueError("key_states is not a tensor")  
            
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
  
        if attention_mask is not None:  
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        if manager.method_name not in manager.not_update:
            if manager.method_name not in manager.last_process:
                # 如果key_states没有压缩，无法进入该函数，正确的应该是 raw_key_states_shape != kv_seq_len 目前跑的版本都是该版本，有少量样本有错
                if past_key_value is not None and key_states.shape[-2] != kv_seq_len:
                    self.kv_cluster.update_kv_generate(past_key_value, attn_weights, key_states, value_states, self.layer_idx, cache_kwargs)
            elif past_key_value is not None and key_states.shape[-2] != kv_seq_len: 
                # print("haha")
                self.kv_cluster.update_all_past_key_values(2, past_key_value, key_states, value_states, self.layer_idx, attn_weights) 
        
        if raw_key_states_shape == kv_seq_len:
            # prifill stage
            if manager.method_name in manager.head_granularity:
                if manager.method_name in manager.layer_differ_recent_n:
                    self.kv_cluster.window_size = manager.layer_window[self.layer_idx]
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attn_weights[:,:,-self.kv_cluster.window_size:,:],attn_weights)
                self.kv_cluster.update_past_key_value(past_key_value, key_states_compress, value_states_compress, self.layer_idx, 0)
            elif manager.method_name in  manager.last_process or manager.method_name in manager.search: #最后一层处理
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attn_weights,None)
                if self.layer_idx == (num_hidden_layers-1):
                    self.kv_cluster.update_all_past_key_values(0, past_key_value, key_states, value_states, None, None)
            else:
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attn_weights, attn_weights)
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
     
        
        attn_output = torch.matmul(attn_weights, value_states)

    if raw_key_states_shape == kv_seq_len and manager.method_name in manager.hidden_delete_stage_and_ours:
        manager.last_attn = attn_weights.sum(0).sum(0)[-8:].sum(0)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    
    return attn_output, attn_weights, past_key_value

def prepare_inputs_for_generation_llama(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    # logger.debug(f"attention_mask.shape: {attention_mask.shape}")
    
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            # cache_length = past_length = past_key_values[0][0].shape[2]
            # max_cache_length = None
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    # logger.debug(f"position_ids: {position_ids}")
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]
    # logger.debug(f"position_ids: {position_ids}")
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

