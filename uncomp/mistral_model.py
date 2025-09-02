import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import (
    logging,
    is_flash_attn_2_available,
)
from uncomp.uncomp_utils import init_uncomp

from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import time


def MistralDecoderLayer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        manager=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
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
    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if self.self_attn.kv_cluster.manager.last_attn!=None and self.self_attn.revise:
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

def mistral_attn_forward_Uncomp(
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
    if manager.method_name in manager.hidden_delete_stage_and_ours and hidden_states.shape[-2] != 1:
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
    bsz, q_len, _ = hidden_states.size()
    manager.q_len = q_len
    
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
        if hasattr(self, "kv_seq_len"): #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
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
            else:          
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    
    # if manager.method_name in manager.head_granularity and raw_key_states_shape != kv_seq_len :
    #     if not isinstance(key_states, list) and manager.method_name not in manager.delet_head_set:
    #         raise ValueError("key_states is not a list")
    #     attn_weights,attn_outputs = [], []
    #     if manager.method_name in manager.delet_head_set:
    #         max_similarity_indices = self.kv_cluster.similarity
    #         attn_weights = torch.matmul(query_states[:,self.kv_cluster.head_pattern,:,:], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    #         if attention_mask is not None:  
    #             causal_mask = attention_mask[:,:, :, : key_states.shape[-2]]
    #             attn_weights = attn_weights + causal_mask
                
    #         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #         attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    #         attn_output = torch.zeros(bsz, self.num_heads, q_len, self.head_dim, dtype=query_states.dtype, device=query_states.device)
    #         attn_output_1 = torch.matmul(attn_weights, value_states)
    #         attn_output[:,self.kv_cluster.head_pattern,:,:] = attn_output_1
    #         attn_output[:,self.kv_cluster.head_indices1[:num_attention_heads-manager.select_topk],:,:] = attn_output_1[:,max_similarity_indices,:,:]

    #         if past_key_value is not None:
    #             self.kv_cluster.update_head_kv_generate(past_key_value, attn_weights, key_states, value_states, self.layer_idx, cache_kwargs)         
    #     else:
    #         for i, (key_state,value_state) in enumerate(zip(key_states,value_states)):
    #             query_state = query_states[:,self.kv_cluster.head_pattern[i],:,:]
    #             attn_weight = torch.matmul(query_state, key_state.transpose(2, 3)) / math.sqrt(self.head_dim)
    #             if attention_mask is not None:  # no matter the length, we just slice it
    #                 causal_mask = attention_mask[:, :, :, : key_state.shape[-2]]
    #                 attn_weight = attn_weight + causal_mask
    #             attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #             attn_weights.append(attn_weight)             
    #             attn_weight = nn.functional.dropout(attn_weight, p=self.attention_dropout, training=self.training)
    #             attn_output_item = torch.matmul(attn_weight, value_state)
    #             attn_outputs.append(attn_output_item)

    #         attn_output = torch.cat(attn_outputs,dim=1)
    #         for i in range(len(key_states)):
    #             if len(self.kv_cluster.head_pattern[i]) != 0:
    #                 attn_output[:,self.kv_cluster.head_pattern[i],:,:] = attn_outputs[i]

    #         if past_key_value is not None:
    #             self.kv_cluster.update_head_kv_generate(past_key_value, attn_weights, key_states, value_states, self.layer_idx, cache_kwargs)   
    # elif manager.method_name in manager.chai and self.layer_idx >= num_hidden_layers//2:
    #     cluster_assignment_log_per_example = dict()
    #     if raw_key_states_shape == kv_seq_len:
    #         num_examples, num_org_heads, seq_len, head_dim = query_states.shape
    #         query_states_four = query_states[:, :, :5, :]
    #         key_states_four = key_states[:, :, :5, :]
    #         scores_four = F.softmax((torch.matmul(query_states_four, key_states_four.transpose(2, 3))/ math.sqrt(self.head_dim)).float(),dim=-1)
    #         scores_four_numpy = scores_four.cpu().numpy()
    #         scores_new_xk_xq = torch.zeros(
    #             [num_examples, num_org_heads, seq_len, seq_len],
    #             device=query_states.device,
    #             dtype=query_states.dtype,
    #         )
    #         xk_new = torch.zeros(
    #             [num_examples, manager.chai_layers[self.layer_idx], seq_len, head_dim],
    #             dtype=key_states.dtype,
    #             device=key_states.device,
    #         )
    #         xq_new = torch.zeros(
    #             [num_examples, manager.chai_layers[self.layer_idx], seq_len, head_dim],
    #             dtype=query_states.dtype,
    #             device=query_states.device,
    #         )
    #         for ex_id in range(num_examples):
    #             assert num_examples == 1
    #             temp_data = dict()
    #             ex_id_score = scores_four_numpy[ex_id, :]
    #             sequence_length_example = ex_id_score.shape[1]
                
    #             num_heads = ex_id_score.shape[0]
    #             first_sample_score = ex_id_score.reshape((num_heads, -1))
    #             dist_arr = cdist(
    #                 first_sample_score, first_sample_score, metric="cosine"
    #             )
    #             cluster = AgglomerativeClustering(
    #                     n_clusters=manager.chai_layers[self.layer_idx],
    #                     metric="precomputed",
    #                    linkage="average",
    #             )
    #             cluster = cluster.fit(dist_arr)
                
    #             cluster_assignment = cluster.labels_
    #             self.grouping = cluster_assignment
    #             for cluster_idx in range(manager.chai_layers[self.layer_idx]):
    #                 grouped_heads = np.where(cluster_assignment == cluster_idx)[
    #                     0
    #                 ].tolist()
    #                 xk_new[ex_id, cluster_idx, :, :] = key_states[ex_id, grouped_heads[0], :, :]
    #                 xq_new[ex_id, cluster_idx, :, :] = query_states[ex_id, grouped_heads[0], :, :]
    #                 temp_data[cluster_idx] = grouped_heads
    #             cluster_assignment_log_per_example[ex_id] = temp_data
    #     else:
    #         num_examples, num_org_heads, seq_len, head_dim = key_states.shape
    #         scores_new_xk_xq = torch.zeros(
    #             [num_examples, num_org_heads, 1, seq_len],
    #             device=query_states.device,
    #             dtype=query_states.dtype,
    #         )
    #         xk_new = torch.zeros(
    #             [num_examples, manager.chai_layers[self.layer_idx], seq_len, head_dim],
    #             dtype=key_states.dtype,
    #             device=key_states.device,
    #         )
    #         xq_new = torch.zeros(
    #             [num_examples, manager.chai_layers[self.layer_idx], 1, head_dim],
    #             dtype=query_states.dtype,
    #             device=query_states.device,
    #         )
    #         cluster_assignment = self.grouping
    #         for ex_id in range(num_examples):
    #             temp_data = dict()
    #             for cluster_idx in range(manager.chai_layers[self.layer_idx]):
    #                 grouped_heads = np.where(cluster_assignment == cluster_idx)[
    #                     0
    #                 ].tolist()
    #                 xk_new[ex_id, cluster_idx, :, :] = key_states[
    #                     ex_id, grouped_heads[0], :, :
    #                 ]
    #                 xq_new[ex_id, cluster_idx, :, :] = query_states[
    #                     ex_id, grouped_heads[0], :, :
    #                 ]
    #                 temp_data[cluster_idx] = grouped_heads
    #             cluster_assignment_log_per_example[ex_id] = temp_data

    #     scores_new_temp = torch.matmul(xq_new, xk_new.transpose(2, 3)) / math.sqrt(self.head_dim)
    #     for ex_id in range(num_examples):
    #         for cluster_idx in range(manager.chai_layers[self.layer_idx]):
    #             scores_new_xk_xq[
    #                 ex_id,
    #                 cluster_assignment_log_per_example[ex_id][cluster_idx],
    #                 :,
    #                 :,
    #             ] = scores_new_temp[ex_id, cluster_idx, :, :]
        
    #     if attention_mask is not None:  # no matter the length, we just slice it
    #         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #         scores_new_xk_xq = scores_new_xk_xq + attention_mask
            
    #     scores_new_xk_xq = F.softmax(scores_new_xk_xq.float(), dim=-1).type_as(query_states)
    #     attn_weights = scores_new_xk_xq
        
    #     attn_output = torch.matmul(attn_weights, value_states)  
    #     if raw_key_states_shape == kv_seq_len:
    #         key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attn_weights, self.num_key_value_groups)
    #         if manager.method_name in manager.head_granularity:
    #             self.kv_cluster.update_past_key_value(past_key_value, key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
    #         else:
    #             past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)               
    # else:
    #     if not isinstance(key_states, torch.Tensor):
    #         print("type(key_states):",type(key_states))
    #         raise ValueError("key_states is not a tensor")  
    #     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    #     if attention_mask is not None:  
    #         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #         attn_weights = attn_weights + causal_mask

    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    #     if past_key_value is not None and key_states.shape[-2] != kv_seq_len:
    #         self.kv_cluster.update_kv_generate(past_key_value, attn_weights, key_states, value_states, self.layer_idx, cache_kwargs)
    #     elif past_key_value is not None and raw_key_states_shape != kv_seq_len and manager.method_name in manager.mask_implement:
    #         self.kv_cluster.update_kv_generate(past_key_value, attn_weights, key_states, value_states, self.layer_idx, cache_kwargs)

    #     if raw_key_states_shape == kv_seq_len:
    #         if manager.method_name in manager.head_granularity:
    #             key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attn_weights[:,:,-self.kv_cluster.window_size:,:], self.num_key_value_groups)
    #             self.kv_cluster.update_past_key_value(past_key_value, key_states_compress, value_states_compress, self.layer_idx, 0)
    #         else:
    #             key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attn_weights, self.num_key_value_groups)
    #             past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
    #     attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    #     attn_output = torch.matmul(attn_weights, value_states)
    
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


    if raw_key_states_shape == kv_seq_len and  manager.method_name in manager.hidden_delete_stage_and_ours:
        manager.last_attn = attn_weights.sum(0).sum(0)[-8:].sum(0)

    
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def prepare_inputs_for_generation_mistral(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    # Omit tokens covered by past_key_values
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
            
            # TODO
            
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    # assert 1==0
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]
    
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

