from typing import Any, Dict, Optional, Tuple,List
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import csv
from accelerate import Accelerator
from scipy import stats
from scipy.spatial.distance import cosine
accelerator = Accelerator()
import sys
pi = math.pi
from uncomp.utils.logger import Logger

logger = Logger(accelerator=accelerator,log_file=None)

from uncomp.utils.entropy_utils import (
    get_entropy_3_dimensions_single_batch,
    get_entropy_4_dimensions_single_batch,
    get_entropy_head_4_dimensions_single_batch,
    get_entropy_5_dimensions_single_batch,
    get_entropy_head_5_dimensions_single_batch,
    get_entropy_head_5_svd_dimensions_single_batch,
    get_head_pattern_attn_entropy,
    get_head_pattern_attn_variance,
    get_entropy_svd_4_dimensions_single_batch,
    get_entropy_head_svd_4_dimensions_single_batch,
    
)
from itertools import accumulate
class UncompCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 8, max_capacity_prompt = 256 + 64, 
                 kernel_size = 5, pooling = 'avgpool', beta = 20,  layer_idx=None ,
                 manager = None):
        self.manager = manager
        self.bsz = manager.bsz 
        self.layer_idx = layer_idx
        self.num_hidden_layers = manager.num_hidden_layers
        self.num_attention_heads = manager.num_attention_heads
        self.window_size = window_size
            
        if manager.method_name in manager.delet_head_set:
            self.select_topk = manager.select_topk 
            self.head_indices1 = manager.head_datas[self.layer_idx]
            self.head_indices11 = torch.cat([manager.head_datas[self.layer_idx][:self.num_hidden_layers-self.select_topk].sort()[0],manager.head_datas[self.layer_idx][-self.select_topk:].sort()[0]],dim=0)
            
            self.recent_indices_generate=[torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.select_topk ,-1)]
            if manager.method_name == "delete_prefill_and_generate_head":
                self.similarity = manager.similar_data[self.layer_idx]
        elif "random" in manager.method_name: 
            self.head_indices1 = torch.randperm(self.num_attention_heads,device='cuda')
            self.head_indices1 = torch.cat([self.head_indices1[:self.num_attention_heads//2].sort()[0],self.head_indices1[self.num_attention_heads//2:].sort()[0]],dim=0)
            self.recent_indices = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers,-1)]
            self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_attention_heads//2,-1),torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers//2,-1)]
        elif "group32" in manager.method_name:
            self.head_indices1 = manager.head_datas[self.layer_idx]
            self.recent_indices = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers,-1)]
            self.recent_indices_generate = [
                torch.arange(-self.window_size, 0, device='cuda').view(1, 1, -1).expand(self.bsz, 1, -1)
                for _ in range(32)
            ]
            manager.num_kv_groups = 32
            
        elif "group3" in manager.method_name:
            self.head_indices1 = manager.head_datas[self.layer_idx]
            self.recent_indices = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers,-1)]
            self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,11,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,10,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,11,-1)
                                            ]
        elif "group4" in manager.method_name:
            self.head_indices1 = manager.head_datas[self.layer_idx]
            self.recent_indices = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers,-1)]
            self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,8,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,8,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,8,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,8,-1),
                                            ]
        elif "group5" in manager.method_name:
            self.head_indices1 = manager.head_datas[self.layer_idx]
            self.recent_indices = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers,-1)]
            self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,7,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,6,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,6,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,6,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,7,-1)
                                            ]
        elif "group8" in manager.method_name:
            self.head_indices1 = manager.head_datas[self.layer_idx]
            self.recent_indices = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers,-1)]
            self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,4,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,4,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,4,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,4,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,4,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,4,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,4,-1),
                                            torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,4,-1),
                                            ]
            manager.num_kv_groups = 8
            manager.streams = [torch.cuda.Stream() for _ in range(manager.num_kv_groups)]
            # manager.events = [torch.cuda.Event() for _ in range(manager.num_kv_groups)]
        else:
            self.head_indices1 = manager.head_datas[self.layer_idx]
            self.recent_indices = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers,-1)]
            self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_attention_heads//2,-1),torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_attention_heads//2,-1)]
            manager.streams = [torch.cuda.Stream() for _ in range(2)]
            # manager.events = [torch.cuda.Event() for _ in range(2)]
            manager.num_kv_groups = 2
            
        self.steps = -1
        self.beta = beta
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def get_memory_info(self):
        stats = torch.cuda.memory_stats()
        reserved = stats.get("reserved_bytes.all.current", 0)
        allocated = stats.get("allocated_bytes.all.current", 0)
        return reserved, allocated

    def bytes_to_gb(self,x):
        return round(x / (1024 ** 3), 4)
                
    
    def update_kv(self, key_states, query_states, value_states, attn_weights_now, attn_weights_now_all):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        manager = self.manager
        num_hidden_layers = manager.num_hidden_layers
        num_attention_heads = manager.num_attention_heads
        num_group_heads_num = num_attention_heads//2
        method_name = self.manager.method_name
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size 
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
        steps = (max_num - min_num) // self.num_hidden_layers 
        max_capacity_prompt = max_num - self.layer_idx * steps

        if "Q" in method_name:
            states = query_states
            my_type = "query"
        elif "K" in method_name:
            states = key_states 
            my_type = "key"
        elif "V" in method_name:
            states = value_states
            my_type = "value"
        else:
            states= query_states
        if True:
            if "streamingllm" in method_name:
                assert key_states.shape[-2] == query_states.shape[-2]
                bsz, num_heads, q_len, head_dim = query_states.shape
                self.window_size = self.max_capacity_prompt // 7
                self.max_capacity_prompt = self.window_size * 6
                # print(f"StreamingLLM max_capacity_prompt {self.max_capacity_prompt}")
                
                if q_len < self.max_capacity_prompt:
                    return key_states, value_states
                else:
                    indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
                    indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                    k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                    v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                    k_cur = key_states[:, :, -self.window_size:, :]
                    v_cur = value_states[:, :, -self.window_size:, :]
                    key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                    value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                    return key_states, value_states
            # search head type
            elif method_name == "head_type_search_2":
                attn_weights = attn_weights_now
                select_topk = num_group_heads_num
                svdn = 32 
                head_pattern = get_head_pattern_attn_entropy(attn_weights,query_states,0,svdn,0,[0,select_topk])
                svdn = "svd" + str(svdn)
                filename = f"./search/512/llama2-chat/query/{svdn}/"
                filename = f"./search/llama3-instruct/2_groups/{svdn}/"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                filename = filename + "head_type_search_layer" + str(self.layer_idx) + ".csv"
                if manager.sample_time == 0:
                    mode = 'w'
                else:
                    mode = 'a'
                with open(filename, mode, newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(head_pattern.to(torch.int8).tolist())
                return key_states, value_states
            elif method_name == "head_type_search_4":
                attn_weights = attn_weights_now
                group_num = 4
                select_topk = num_attention_heads//group_num
                svdn = 32
                entropy = get_head_pattern_attn_entropy(attn_weights,query_states,0,svdn,0,[1,select_topk])
                sorted_indices = torch.argsort(entropy)
                labels = torch.empty_like(entropy, dtype=torch.long)
                labels[sorted_indices[:8]] = 0   
                labels[sorted_indices[8:16]] = 1  
                labels[sorted_indices[16:24]] = 2    
                labels[sorted_indices[24:]] = 3
                
                svdn = "svd" + str(svdn)
                filename = f"./search/llama3-instruct/4_groups/{svdn}/"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                filename = filename + "head_type_search_layer" + str(self.layer_idx) + ".csv"
                if manager.sample_time == 0:
                    mode = 'w'
                else:
                    mode = 'a'
                with open(filename, mode, newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(labels.to(torch.int8).tolist())
                return key_states, value_states
            elif method_name == "head_type_search_8":
                attn_weights = attn_weights_now
                group_num = 8
                select_topk = num_attention_heads//group_num
                svdn = 32
                entropy = get_head_pattern_attn_entropy(attn_weights,query_states,0,svdn,0,[1,select_topk])
                sorted_indices = torch.argsort(entropy)
                labels = torch.empty_like(entropy, dtype=torch.long)
                for i in range(group_num):
                    labels[sorted_indices[0+select_topk*i:select_topk*(i+1)]] = i

                svdn = "svd" + str(svdn)
                filename = f"./search/mistral/llama2-chat-13B/8_groups/{svdn}/"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                filename = filename + "head_type_search_layer" + str(self.layer_idx) + ".csv"
                if manager.sample_time == 0:
                    mode = 'w'
                else:
                    mode = 'a'
                with open(filename, mode, newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(labels.to(torch.int8).tolist())
                return key_states, value_states
            elif method_name == "head_type_search_32":
                attn_weights = attn_weights_now
                group_num = 32
                select_topk = num_attention_heads//group_num
                svdn = 32
                entropy = get_head_pattern_attn_entropy(attn_weights,query_states,0,svdn,0,[1,select_topk])
                sorted_indices = torch.argsort(entropy)
                labels = torch.empty_like(entropy, dtype=torch.long)
                for i in range(32):
                    labels[sorted_indices[i:i+1]] = i

                svdn = "svd" + str(svdn)
                filename = f"./search/llama3-instruct/32_groups/{svdn}/"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                filename = filename + "head_type_search_layer" + str(self.layer_idx) + ".csv"
                if manager.sample_time == 0:
                    mode = 'w'
                else:
                    mode = 'a'
                with open(filename, mode, newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(labels.to(torch.int8).tolist())
                return key_states, value_states
            
            elif method_name in manager.ahead_500_equal_code: 
                max_capacity_prompt = self.max_capacity_prompt
                if q_len < max_capacity_prompt:
                    max_capacity_prompt = q_len
                if self.window_size > q_len:
                    self.window_size = q_len
                if self.window_size > max_capacity_prompt//2:
                    self.window_size = max_capacity_prompt//2
                    self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_attention_heads//2,-1),torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers//2,-1)]
                if self.max_capacity_prompt == 86:
                    if "new1" in method_name:
                        max_capacity_prompts = [32,96]
                    elif "new2" in method_name:
                        max_capacity_prompts = [96,32]
                    elif "new3" in method_name:
                        max_capacity_prompts = [16,112]
                    else:
                        max_capacity_prompts = [max_capacity_prompt//2,max_capacity_prompt]
                    max_capacity_prompts = [16,112]
                else:
                    max_capacity_prompt = max_capacity_prompt * 3 // 2 
                    max_capacity_prompts = [max_capacity_prompt//2,max_capacity_prompt]
                    # max_capacity_prompts = [128,640]
                    # max_capacity_prompts = [240,528]
                    if q_len <= max_capacity_prompt:
                        max_capacity_prompts = [q_len//2,q_len]
                max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                if self.layer_idx == num_hidden_layers-1:
                    print("max_capacity_prompts",max_capacity_prompts)
                attn_weights = attn_weights_now
                attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')

                top_k = max_capacity_prompts[1] - self.window_size
                indices = attn_cache.topk(top_k, dim=-1).indices
                indices1 = indices.sort(dim=-1).values
                for i in range(len(self.recent_indices_generate)):
                    self.recent_indices_generate[i] = self.recent_indices_generate[i].to(indices1.device)
                self.head_indices1 = self.head_indices1.to(indices1.device)
                recent_indices = self.recent_indices_generate[0]+q_len
                num_heads = num_attention_heads // 2
                indices_1 = torch.cat([indices1[:,self.head_indices1[-num_heads:],:],recent_indices],dim=-1)
                indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                revise_key_states=key_states[:,self.head_indices1[-num_heads:],:,:].gather(dim = 2, index = indices_expanded)
                revise_value_states=value_states[:,self.head_indices1[-num_heads:],:,:].gather(dim = 2, index = indices_expanded)
                self.attn_weights_1=attn_weights[:, self.head_indices1[-num_heads:], :,:].gather(dim = -1, index = indices_attn)

                max_capacity_prompt_2 = max_capacity_prompts[0]
                top_k2 = max_capacity_prompt_2 - self.window_size
                indices = attn_cache.topk(top_k2, dim=-1).indices
                indices_2 = indices.sort(dim=-1).values
                indices_2 = torch.cat([indices_2[:,self.head_indices1[:num_heads],:],recent_indices],dim=-1)
                indices_expanded_2  = indices_2.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices_attn_2 = indices_2.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                revise_value_states_2 = value_states[:,self.head_indices1[:num_heads],:,:].gather(dim=2,index=indices_expanded_2)
                revise_key_states_2 = key_states[:,self.head_indices1[:num_heads],:,:].gather(dim=2, index=indices_expanded_2)
                self.attn_weights_2 = attn_weights[:, self.head_indices1[:num_heads], :,:].gather(dim = -1, index = indices_attn_2)
                key1 = revise_key_states
                key2 = revise_key_states_2
                value1 = revise_value_states
                value2 = revise_value_states_2
                revise_key_states = [key1, key2]
                revise_value_states = [value1, value2]
                self.head_pattern = [self.head_indices1[-num_heads:], self.head_indices1[:num_heads]]
                self.attn_weights = [self.attn_weights_1, self.attn_weights_2]
                self.cache_size = [max_capacity_prompts[1], max_capacity_prompts[0]]     

                get_memory_info = self.get_memory_info
                bytes_to_gb = self.bytes_to_gb
                if self.layer_idx == 0:
                    torch.cuda.empty_cache()
                    manager.before_reserved, manager.before_allocated = get_memory_info()
                del key_states, value_states, attn_weights_now, attn_weights_now_all
                torch.cuda.empty_cache()
                
                if self.layer_idx == num_hidden_layers - 1:
                    torch.cuda.empty_cache()
                    end_reserved, end_allocated = get_memory_info()
                    # print(f"Memory usage before: reserved={bytes_to_gb(manager.before_reserved)} GB, allocated={bytes_to_gb(manager.before_allocated)} GB")
                    # print(f"Memory usage after:  reserved={bytes_to_gb(end_reserved)} GB, allocated={bytes_to_gb(end_allocated)} GB")
                    # print(f"Memory freed:        reserved={bytes_to_gb(manager.before_reserved - end_reserved)} GB, allocated={bytes_to_gb(manager.before_allocated - end_allocated)} GB")
                    # assert 1 == 0  #
            # extreme_compressibility
            elif method_name in manager.extreme_compressibility_equal_code :
                max_capacity_prompt = self.max_capacity_prompt
                if q_len < max_capacity_prompt:
                    max_capacity_prompt = q_len
                self.cache_size = max_capacity_prompt
                attn_weights = attn_weights_now
                attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')
                top_k = max_capacity_prompt - self.window_size
                indices = attn_cache.topk(top_k, dim=-1).indices
                indices1 = indices.sort(dim=-1).values
                recent_indices = self.recent_indices_generate[0]+q_len
                indices_1 = torch.cat([indices1[:,self.head_indices1[-num_group_heads_num:],:],recent_indices],dim=-1)
                indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                revise_key_states=key_states[:,self.head_indices1[-num_group_heads_num:],:,:].gather(dim = 2, index = indices_expanded)
                revise_value_states=value_states[:,self.head_indices1[-num_group_heads_num:],:,:].gather(dim = 2, index = indices_expanded)
                self.attn_weights_1=attn_weights[:, self.head_indices1[-num_group_heads_num:], :,:].gather(dim = -1, index = indices_attn)
                
                if "128" in method_name:
                    nums = 4
                elif "64" in method_name:
                    nums = 8
                elif "32" in method_name:
                    nums = 16
                elif "16" in method_name:
                    nums = 32
                elif "12" in method_name:
                    nums = 42
                elif "10" in method_name:
                    nums = 51
                max_capacity_prompt_2 = max_capacity_prompt//nums
                if q_len < max_capacity_prompt_2:
                    max_capacity_prompt = q_len
                if max_capacity_prompt_2 < self.window_size:
                    max_capacity_prompt_2 = self.window_size
                top_k2 = max_capacity_prompt_2 - self.window_size
                indices_2 = indices[:,:,:top_k2].sort(dim=-1).values
                indices_2 = torch.cat([indices_2[:,self.head_indices1[:num_group_heads_num],:],recent_indices],dim=-1)
                indices_expanded_2  = indices_2.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices_attn_2 = indices_2.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                revise_value_states_2 = value_states[:,self.head_indices1[:num_group_heads_num],:,:].gather(dim=2,index=indices_expanded_2)
                revise_key_states_2 = key_states[:,self.head_indices1[:num_group_heads_num],:,:].gather(dim=2, index=indices_expanded_2)
                self.attn_weights_2 = attn_weights[:, self.head_indices1[:num_group_heads_num], :,:].gather(dim = -1, index = indices_attn_2)
                key1 = revise_key_states
                key2 = revise_key_states_2
                value1 = revise_value_states
                value2 = revise_value_states_2
                revise_key_states = [key1, key2]
                revise_value_states = [value1, value2]
                self.head_pattern = [self.head_indices1[-num_group_heads_num:], self.head_indices1[:num_group_heads_num]]
                self.attn_weights = [self.attn_weights_1, self.attn_weights_2]
                self.cache_size = [max_capacity_prompt, max_capacity_prompt_2] 
            elif method_name in manager.delete_head_equal_code:
                max_capacity_prompt = self.max_capacity_prompt
                if q_len < max_capacity_prompt:
                    max_capacity_prompt = q_len
                self.cache_size = max_capacity_prompt
                attn_weights = attn_weights_now
                attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')
                top_k = max_capacity_prompt - self.window_size
                indices = attn_cache.topk(top_k, dim=-1).indices
                recent_indices = self.recent_indices_generate[0]+q_len
                select_topk = self.select_topk
                
                indices_1 = torch.cat([indices[:,self.head_indices1[-select_topk:],:],recent_indices],dim=-1)
                indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                revise_key_states=key_states[:,self.head_indices1[-select_topk:],:,:].gather(dim = 2, index = indices_expanded)
                revise_value_states=value_states[:,self.head_indices1[-select_topk:],:,:].gather(dim = 2, index = indices_expanded)
                self.attn_weights_1=attn_weights[:, self.head_indices1[-select_topk:], :,:].gather(dim = -1, index = indices_attn)
                self.head_pattern = self.head_indices1[-select_topk:]
                self.attn_weights = self.attn_weights_1
                self.cache_size = max_capacity_prompt
                attn_1 = key_states[:,self.head_indices1[:32-select_topk],:,:].squeeze(0)[:,-8:,:].view(32-select_topk,-1)
                attn_2 = key_states[:,self.head_indices1[-select_topk:],:,:].squeeze(0)[:,-8:,:].view(select_topk,-1)
                similarity_matrix = torch.nn.functional.cosine_similarity(attn_1.unsqueeze(1), attn_2.unsqueeze(0), dim=2)
                max_similarity_indices = torch.argmax(similarity_matrix, dim=1)
                self.similarity = max_similarity_indices
            # multi groups
            elif "pyramidkv" not in method_name:
                if method_name in manager.ahead_500_equal_code_group3: 
                    max_capacity_prompt = self.max_capacity_prompt
                    if q_len < max_capacity_prompt:
                        max_capacity_prompt = q_len
                    attn_weights = attn_weights_now
                    attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                    if self.pooling == 'avgpool':
                        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    elif self.pooling == 'maxpool':
                        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    else:
                        raise ValueError('Pooling method not supported')
                    
                    if self.max_capacity_prompt == 86:
                        max_capacity_prompts = [112,64,16]
                        if "new1" in method_name:
                            max_capacity_prompts = [116,64,12]
                    
                    if self.max_capacity_prompt == 512:
                        max_capacity_prompts = [736,384,32]
                        if "new1" in method_name:
                            max_capacity_prompts = [576,384,192]
                        elif "new2" in method_name:
                            max_capacity_prompts = [640,384,128]
                        elif "new3" in method_name:
                            max_capacity_prompts = [704,384,64]
                    
                    max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                    if self.layer_idx == 0:
                        print("max_capacity_prompts",max_capacity_prompts)
                    max_capacity_prompt = max_capacity_prompts[0]
                    top_k = max_capacity_prompt - self.window_size
                    indices = attn_cache.topk(top_k, dim=-1).indices
                    indices1 = indices.sort(dim=-1).values
                    recent_indices = self.recent_indices_generate[0]+q_len
                    indices_1 = torch.cat([indices1[:,self.head_indices1[-11:],:],recent_indices],dim=-1)
                    indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_key_states=key_states[:,self.head_indices1[-11:],:,:].gather(dim = 2, index = indices_expanded)
                    revise_value_states=value_states[:,self.head_indices1[-11:],:,:].gather(dim = 2, index = indices_expanded)
                    attn_weights_1=attn_weights[:, self.head_indices1[-11:], :,:].gather(dim = -1, index = indices_attn)

                    max_capacity_prompt = max_capacity_prompts[1]
                    top_k2 = max_capacity_prompt - self.window_size
                    indices_2 = indices[:,:,:top_k2].sort(dim=-1).values
                    recent_indices_1 = self.recent_indices_generate[1]+q_len
                    indices_2 = torch.cat([indices_2[:,self.head_indices1[11:-11],:],recent_indices_1],dim=-1)
                    indices_expanded_2  = indices_2.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_2 = indices_2.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_2 = value_states[:,self.head_indices1[11:-11],:,:].gather(dim=2,index=indices_expanded_2)
                    revise_key_states_2 = key_states[:,self.head_indices1[11:-11],:,:].gather(dim=2, index=indices_expanded_2)
                    attn_weights_2 = attn_weights[:, self.head_indices1[11:-11], :,:].gather(dim = -1, index = indices_attn_2)
                    
                    max_capacity_prompt = max_capacity_prompts[2]
                    top_k3 = max_capacity_prompt - self.window_size
                    indices_3 = indices[:,:,:top_k3].sort(dim=-1).values
                    indices_3 = torch.cat([indices_3[:,self.head_indices1[:11],:],recent_indices],dim=-1)
                    indices_expanded_3  = indices_3.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_3 = indices_3.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_3 = value_states[:,self.head_indices1[:11],:,:].gather(dim=2,index=indices_expanded_3)
                    revise_key_states_3 = key_states[:,self.head_indices1[:11],:,:].gather(dim=2, index=indices_expanded_3)
                    attn_weights_3 = attn_weights[:, self.head_indices1[:11], :,:].gather(dim = -1, index = indices_attn_3)
                    
                    key1 = revise_key_states
                    key2 = revise_key_states_2
                    key3 = revise_key_states_3
                    value1 = revise_value_states
                    value2 = revise_value_states_2
                    value3 = revise_value_states_3
                    revise_key_states = [key1, key2, key3]
                    revise_value_states = [value1, value2, value3]
                    self.head_pattern = [self.head_indices1[-11:], self.head_indices1[11:-11], self.head_indices1[:11]]
                    self.attn_weights = [attn_weights_1, attn_weights_2, attn_weights_3]
                    self.cache_size = max_capacity_prompts   
                elif method_name in manager.ahead_500_equal_code_group4: 
                    max_capacity_prompt = self.max_capacity_prompt
                    if q_len < max_capacity_prompt:
                        max_capacity_prompt = q_len
                    attn_weights = attn_weights_now
                    attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                    
                    if self.pooling == 'avgpool':
                        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    elif self.pooling == 'maxpool':
                        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    else:
                        raise ValueError('Pooling method not supported')
                    
                    if self.max_capacity_prompt == 86:
                        max_capacity_prompts = [112,80,48,16]
                        if "new1" in method_name:
                            max_capacity_prompts = [116,81,47,12]
                    elif self.max_capacity_prompt == 512:
                        # max_capacity_prompts = [512,427,341,256]
                        max_capacity_prompts = [640,470,298,128]
                        
                    else:
                        max_capacity_prompts = []
                        max_cap = max_capacity_prompt
                        min_cap = max_capacity_prompt // 2
                        allowance = (max_cap-min_cap) // 3
                        for i in range(4):
                            max_capacity_prompts.append(max_cap - allowance*i)
                    max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                    if self.layer_idx == 0:
                        print("max_capacity_prompts",max_capacity_prompts)
                    max_capacity_prompt = max_capacity_prompts[0]
                    top_k = max_capacity_prompt - self.window_size
                    indices = attn_cache.topk(top_k, dim=-1).indices
                    indices1 = indices.sort(dim=-1).values
                    recent_indices = self.recent_indices_generate[0]+q_len
                    indices_1 = torch.cat([indices1[:,self.head_indices1[-8:],:],recent_indices],dim=-1)
                    indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_key_states=key_states[:,self.head_indices1[-8:],:,:].gather(dim = 2, index = indices_expanded)
                    revise_value_states=value_states[:,self.head_indices1[-8:],:,:].gather(dim = 2, index = indices_expanded)
                    attn_weights_1=attn_weights[:, self.head_indices1[-8:], :,:].gather(dim = -1, index = indices_attn)

                    max_capacity_prompt = max_capacity_prompts[1]
                    top_k2 = max_capacity_prompt - self.window_size
                    indices_2 = indices[:,:,:top_k2].sort(dim=-1).values
                    recent_indices = self.recent_indices_generate[1]+q_len
                    recent_indices_1 = self.recent_indices_generate[1]+q_len
                    indices_2 = torch.cat([indices_2[:,self.head_indices1[16:-8],:],recent_indices_1],dim=-1)
                    indices_expanded_2  = indices_2.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_2 = indices_2.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_2 = value_states[:,self.head_indices1[16:-8],:,:].gather(dim=2,index=indices_expanded_2)
                    revise_key_states_2 = key_states[:,self.head_indices1[16:-8],:,:].gather(dim=2, index=indices_expanded_2)
                    attn_weights_2 = attn_weights[:, self.head_indices1[16:-8], :,:].gather(dim = -1, index = indices_attn_2)
                    
                    max_capacity_prompt = max_capacity_prompts[2]
                    top_k3 = max_capacity_prompt - self.window_size
                    recent_indices = self.recent_indices_generate[2]+q_len
                    indices_3 = indices[:,:,:top_k3].sort(dim=-1).values
                    indices_3 = torch.cat([indices_3[:,self.head_indices1[8:16],:],recent_indices],dim=-1)
                    indices_expanded_3  = indices_3.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_3 = indices_3.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_3 = value_states[:,self.head_indices1[8:16],:,:].gather(dim=2,index=indices_expanded_3)
                    revise_key_states_3 = key_states[:,self.head_indices1[8:16],:,:].gather(dim=2, index=indices_expanded_3)
                    attn_weights_3 = attn_weights[:, self.head_indices1[8:16], :,:].gather(dim = -1, index = indices_attn_3)
                    
                    max_capacity_prompt = max_capacity_prompts[3]
                    top_k4 = max_capacity_prompt - self.window_size
                    indices_4 = indices[:,:,:top_k4].sort(dim=-1).values
                    recent_indices = self.recent_indices_generate[3]+q_len
                    indices_4 = torch.cat([indices_4[:,self.head_indices1[:8],:],recent_indices],dim=-1)
                    indices_expanded_4  = indices_4.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_4 = indices_4.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_4 = value_states[:,self.head_indices1[:8],:,:].gather(dim=2,index=indices_expanded_4)
                    revise_key_states_4 = key_states[:,self.head_indices1[:8],:,:].gather(dim=2, index=indices_expanded_4)
                    attn_weights_4 = attn_weights[:, self.head_indices1[:8], :,:].gather(dim = -1, index = indices_attn_4)
                    
                    key1,key2,key3,key4 = revise_key_states,revise_key_states_2,revise_key_states_3,revise_key_states_4
                    value1,value2,value3,value4 = revise_value_states,revise_value_states_2,revise_value_states_3,revise_value_states_4
                    revise_key_states = [key1, key2, key3, key4]
                    revise_value_states = [value1, value2, value3, value4]
                    self.head_pattern = [self.head_indices1[-8:], self.head_indices1[16:-8], self.head_indices1[8:16], self.head_indices1[:8]]
                    self.attn_weights = [attn_weights_1, attn_weights_2, attn_weights_3, attn_weights_4]
                    self.cache_size = max_capacity_prompts  
                elif method_name in manager.ahead_500_equal_code_group5: 
                    max_capacity_prompt = self.max_capacity_prompt
                    if q_len < max_capacity_prompt:
                        max_capacity_prompt = q_len
                    attn_weights = attn_weights_now
                    attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                    if self.pooling == 'avgpool':
                        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    elif self.pooling == 'maxpool':
                        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    else:
                        raise ValueError('Pooling method not supported')
                    if self.max_capacity_prompt == 512:
                        max_capacity_prompts = [512,448,384,320,256]
                        if "new1" in method_name:
                            max_capacity_prompts = [640,512,384,256,128]
                    if self.max_capacity_prompt == 86:
                        max_capacity_prompts = [112,88,64,40,16]    
                        if "new1" in method_name:
                                max_capacity_prompts = [116,90,64,38,12]
                    max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                    if self.layer_idx == 0:
                        print("max_capacity_prompts",max_capacity_prompts)  
                    max_capacity_prompt = max_capacity_prompts[0]
                    top_k = max_capacity_prompt - self.window_size
                    indices = attn_cache.topk(top_k, dim=-1).indices
                    indices1 = indices.sort(dim=-1).values
                    recent_indices = self.recent_indices_generate[0]+q_len
                    indices_1 = torch.cat([indices1[:,self.head_indices1[-7:],:],recent_indices],dim=-1)
                    indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_key_states=key_states[:,self.head_indices1[-7:],:,:].gather(dim = 2, index = indices_expanded)
                    revise_value_states=value_states[:,self.head_indices1[-7:],:,:].gather(dim = 2, index = indices_expanded)
                    attn_weights_1=attn_weights[:, self.head_indices1[-7:], :,:].gather(dim = -1, index = indices_attn)

                    max_capacity_prompt = max_capacity_prompts[1]
                    recent_indices = self.recent_indices_generate[1]+q_len
                    top_k2 = max_capacity_prompt - self.window_size
                    indices_2 = indices[:,:,:top_k2].sort(dim=-1).values
                    recent_indices_1 = self.recent_indices_generate[1]+q_len
                    indices_2 = torch.cat([indices_2[:,self.head_indices1[19:-7],:],recent_indices_1],dim=-1)
                    indices_expanded_2  = indices_2.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_2 = indices_2.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_2 = value_states[:,self.head_indices1[19:-7],:,:].gather(dim=2,index=indices_expanded_2)
                    revise_key_states_2 = key_states[:,self.head_indices1[19:-7],:,:].gather(dim=2, index=indices_expanded_2)
                    attn_weights_2 = attn_weights[:, self.head_indices1[19:-7], :,:].gather(dim = -1, index = indices_attn_2)
                    
                    max_capacity_prompt = max_capacity_prompts[2]
                    recent_indices = self.recent_indices_generate[2]+q_len
                    top_k3 = max_capacity_prompt - self.window_size
                    indices_3 = indices[:,:,:top_k3].sort(dim=-1).values
                    indices_3 = torch.cat([indices_3[:,self.head_indices1[13:19],:],recent_indices],dim=-1)
                    indices_expanded_3  = indices_3.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_3 = indices_3.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_3 = value_states[:,self.head_indices1[13:19],:,:].gather(dim=2,index=indices_expanded_3)
                    revise_key_states_3 = key_states[:,self.head_indices1[13:19],:,:].gather(dim=2, index=indices_expanded_3)
                    attn_weights_3 = attn_weights[:, self.head_indices1[13:19], :,:].gather(dim = -1, index = indices_attn_3)
                    
                    max_capacity_prompt = max_capacity_prompts[3]
                    recent_indices = self.recent_indices_generate[3]+q_len
                    top_k4 = max_capacity_prompt - self.window_size
                    indices_4 = indices[:,:,:top_k4].sort(dim=-1).values
                    indices_4 = torch.cat([indices_4[:,self.head_indices1[7:13],:],recent_indices],dim=-1)
                    indices_expanded_4  = indices_4.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_4 = indices_4.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_4 = value_states[:,self.head_indices1[7:13],:,:].gather(dim=2,index=indices_expanded_4)
                    revise_key_states_4 = key_states[:,self.head_indices1[7:13],:,:].gather(dim=2, index=indices_expanded_4)
                    attn_weights_4 = attn_weights[:, self.head_indices1[7:13], :,:].gather(dim = -1, index = indices_attn_4)
                    
                    max_capacity_prompt = max_capacity_prompts[4]
                    recent_indices = self.recent_indices_generate[4]+q_len
                    top_k5 = max_capacity_prompt - self.window_size
                    indices_5 = indices[:,:,:top_k5].sort(dim=-1).values
                    indices_5 = torch.cat([indices_5[:,self.head_indices1[:7],:],recent_indices],dim=-1)
                    indices_expanded_5  = indices_5.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_5 = indices_5.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_5 = value_states[:,self.head_indices1[:7],:,:].gather(dim=2,index=indices_expanded_5)
                    revise_key_states_5 = key_states[:,self.head_indices1[:7],:,:].gather(dim=2, index=indices_expanded_5)
                    attn_weights_5 = attn_weights[:, self.head_indices1[:7], :,:].gather(dim = -1, index = indices_attn_5)
                    
                    key1,key2,key3,key4,key5 = revise_key_states,revise_key_states_2,revise_key_states_3,revise_key_states_4,revise_key_states_5
                    value1,value2,value3,value4,value5 = revise_value_states,revise_value_states_2,revise_value_states_3,revise_value_states_4,revise_value_states_5
                    revise_key_states = [key1, key2, key3, key4, key5]
                    revise_value_states = [value1, value2, value3, value4, value5]
                    self.head_pattern = [self.head_indices1[-7:], self.head_indices1[19:-7], self.head_indices1[13:19], self.head_indices1[7:13], self.head_indices1[:7]]
                    self.attn_weights = [attn_weights_1, attn_weights_2, attn_weights_3, attn_weights_4, attn_weights_5]
                    self.cache_size = max_capacity_prompts         
                elif method_name in manager.ahead_500_equal_code_group8: 
                    if num_attention_heads == 32:
                        max_capacity_prompt = self.max_capacity_prompt
                        manager.layer_window = [8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8]
                        self.window_size = manager.layer_window[self.layer_idx]
                        self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_attention_heads//8,-1)]*8
                        if q_len < max_capacity_prompt:
                            max_capacity_prompt = q_len
                        attn_weights = attn_weights_now_all
                        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                        if self.pooling == 'avgpool':
                            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        elif self.pooling == 'maxpool':
                            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        else:
                            raise ValueError('Pooling method not supported')
                        
                        
                        if self.max_capacity_prompt == 512 and "new" in method_name :
                            # max_capacity_prompts = [32,132,232,332,436,536,636,736]
                            max_capacity_prompts = [32,132,232,334,434,534,635,736]
                            if "new1" in method_name:
                                max_capacity_prompts = [129,202,275,348,421,494,567,640]
                            elif "new2" in method_name:
                                max_capacity_prompts = [64,158,249,340,431,522,613,704]
                            elif "new3" in method_name:
                                max_capacity_prompts = [32,132,232,334,434,534,635,736]
                            elif "new4" in method_name:
                                max_capacity_prompts = [320,340,358,376,394,412,430,448]
                        elif self.max_capacity_prompt == 86:
                            max_capacity_prompts = [16,30,44,64,64,84,98,112]
                            if "new1" in method_name:
                                max_capacity_prompts = [12,27,42,56,72,86,101,116]
                        else:
                            max_capacity_prompts = []
                            max_cap = max_capacity_prompt
                            min_cap = max_capacity_prompt // 2
                            allowance = (max_cap-min_cap) // 7
                            for i in range(7,-1,-1):
                                max_capacity_prompts.append(max_cap - allowance*i)

                            # max_capacity_prompts = [341,248,155,64,704,613,520,427]
                            # max_capacity_prompts = [341,248,155,64,427,520,613,704]
                            # max_capacity_prompts = [347,274,201,128,640,494,567,421]
                            max_capacity_prompts = [20,128,256,384,384,512,640,748]
                        max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                        
                        revise_key_states = []
                        revise_value_states = []
                        self.head_pattern = []
                        self.attn_weights = []
                        if self.layer_idx == 0:
                            logger.info("max_capacity_prompts:{}".format(max_capacity_prompts))
                            print("max_capacity_prompts",max_capacity_prompts)
                        new_max_capacity_prompts = []
                        for i in range(0,8,1):
                            max_capacity_prompt = max_capacity_prompts[i]
                            new_max_capacity_prompts.append(max_capacity_prompt)
                            top_k = max_capacity_prompt - self.window_size
                            indices = attn_cache.topk(top_k, dim=-1).indices
                            indices1 = indices.sort(dim=-1).values
                            recent_indices = self.recent_indices_generate[i]+q_len
                            indices_1 = torch.cat([indices1[:,self.head_indices1[4*i:4*(i+1)],:],recent_indices],dim=-1)
                            indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                            indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                            revise_key_states_single=key_states[:,self.head_indices1[4*i:4*(i+1)],:,:].gather(dim = 2, index = indices_expanded)
                            revise_value_states_single=value_states[:,self.head_indices1[4*i:4*(i+1)],:,:].gather(dim = 2, index = indices_expanded)
                            attn_weights_1=attn_weights[:, self.head_indices1[4*i:4*(i+1)], :,:].gather(dim = -1, index = indices_attn)

                            revise_key_states.append(revise_key_states_single)
                            revise_value_states.append(revise_value_states_single)
                            self.attn_weights.append(attn_weights_1)
                            self.head_pattern.append(self.head_indices1[4*i:4*(i+1)])
                        self.cache_size = new_max_capacity_prompts
                    else:
                        max_capacity_prompt = self.max_capacity_prompt
                        manager.layer_window = [8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8,
                                                8,8,8,8]
                        self.window_size = manager.layer_window[self.layer_idx]
                        self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_attention_heads//8,-1)]*8
                        if q_len < max_capacity_prompt:
                            max_capacity_prompt = q_len
                        attn_weights = attn_weights_now_all
                        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                        if self.pooling == 'avgpool':
                            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        elif self.pooling == 'maxpool':
                            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        else:
                            raise ValueError('Pooling method not supported')
                        
                        
                        if self.max_capacity_prompt == 512 and "new" in method_name :
                            # max_capacity_prompts = [32,132,232,332,436,536,636,736]
                            max_capacity_prompts = [32,132,232,334,434,534,635,736]
                            if "new1" in method_name:
                                max_capacity_prompts = [129,202,275,348,421,494,567,640]
                            elif "new2" in method_name:
                                max_capacity_prompts = [64,158,249,340,431,522,613,704]
                            elif "new3" in method_name:
                                max_capacity_prompts = [32,132,232,334,434,534,635,736]
                            elif "new4" in method_name:
                                max_capacity_prompts = [320,340,358,376,394,412,430,448]
                        elif self.max_capacity_prompt == 86:
                            max_capacity_prompts = [16,30,44,64,64,84,98,112]
                            if "new1" in method_name:
                                max_capacity_prompts = [12,27,42,56,72,86,101,116]
                        else:
                            max_capacity_prompts = []
                            max_cap = max_capacity_prompt
                            min_cap = max_capacity_prompt // 2
                            allowance = (max_cap-min_cap) // 7
                            for i in range(7,-1,-1):
                                max_capacity_prompts.append(max_cap - allowance*i)

                            # max_capacity_prompts = [341,248,155,64,704,613,520,427]
                            # max_capacity_prompts = [341,248,155,64,427,520,613,704]
                            # max_capacity_prompts = [347,274,201,128,640,494,567,421]
                            max_capacity_prompts = [20,128,256,384,384,512,640,748]
                        max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                        
                        revise_key_states = []
                        revise_value_states = []
                        self.head_pattern = []
                        self.attn_weights = []
                        if self.layer_idx == 0:
                            logger.info("max_capacity_prompts:{}".format(max_capacity_prompts))
                            print("max_capacity_prompts",max_capacity_prompts)
                        new_max_capacity_prompts = []
                        for i in range(0,8,1):
                            max_capacity_prompt = max_capacity_prompts[i]
                            new_max_capacity_prompts.append(max_capacity_prompt)
                            top_k = max_capacity_prompt - self.window_size
                            indices = attn_cache.topk(top_k, dim=-1).indices
                            indices1 = indices.sort(dim=-1).values
                            recent_indices = self.recent_indices_generate[i]+q_len
                            indices_1 = torch.cat([indices1[:,self.head_indices1[5*i:5*(i+1)],:],recent_indices],dim=-1)
                            indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                            indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                            revise_key_states_single=key_states[:,self.head_indices1[5*i:5*(i+1)],:,:].gather(dim = 2, index = indices_expanded)
                            revise_value_states_single=value_states[:,self.head_indices1[5*i:5*(i+1)],:,:].gather(dim = 2, index = indices_expanded)
                            attn_weights_1=attn_weights[:, self.head_indices1[5*i:5*(i+1)], :,:].gather(dim = -1, index = indices_attn)

                            revise_key_states.append(revise_key_states_single)
                            revise_value_states.append(revise_value_states_single)
                            self.attn_weights.append(attn_weights_1)
                            self.head_pattern.append(self.head_indices1[5*i:5*(i+1)])
                        self.cache_size = new_max_capacity_prompts  
                elif method_name in manager.ahead_500_equal_code_group32:
                    max_capacity_prompt = self.max_capacity_prompt
                    if q_len < max_capacity_prompt:
                        max_capacity_prompt = q_len
                    attn_weights = attn_weights_now
                    attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                    if self.pooling == 'avgpool':
                        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    elif self.pooling == 'maxpool':
                        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    else:
                        raise ValueError('Pooling method not supported')

                    capacity = max_capacity_prompt*3//2
                    steps = capacity // 33
                    max_capacity_prompts = []
                    # if steps < self.window_size:
                    #     first = self.window_size + 2
                    #     end = capacity-first
                    #     steps = (end-first)//31
                    
                    #     for i in range (32):
                    #         max_capacity_prompts.append(int(first+steps*i))
                    # else:
                    #     for i in range (32):
                    #         max_capacity_prompts.append(int(steps*(i+1)))
                    
                    min_num = 128
                    max_num = 640
                    steps = (max_num - min_num) // 31
                    for i in range(32):
                        max_capacity_prompts.append(max_num - steps*(31-i))
                    
                    max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                    
                    revise_key_states = []
                    revise_value_states = []
                    self.head_pattern = []
                    self.attn_weights = []
                    
                    new_max_capacity_prompts = []
                    for i in range(31,-1,-1):
                        max_capacity_prompt = max_capacity_prompts[i]
                        new_max_capacity_prompts.append(max_capacity_prompt)
                        top_k = max_capacity_prompt - self.window_size
                        indices = attn_cache.topk(top_k, dim=-1).indices
                        indices1 = indices.sort(dim=-1).values
                        recent_indices = self.recent_indices_generate[i]+q_len
                        indices_1 = torch.cat([indices1[:,self.head_indices1[i:i+1],:],recent_indices],dim=-1)
                        indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                        indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                        revise_key_states_single=key_states[:,self.head_indices1[i:i+1],:,:].gather(dim = 2, index = indices_expanded)
                        revise_value_states_single=value_states[:,self.head_indices1[i:i+1],:,:].gather(dim = 2, index = indices_expanded)
                        attn_weights_1=attn_weights[:, self.head_indices1[i:i+1], :,:].gather(dim = -1, index = indices_attn)

                        revise_key_states.append(revise_key_states_single)
                        revise_value_states.append(revise_value_states_single)
                        self.attn_weights.append(attn_weights_1)
                        self.head_pattern.append(self.head_indices1[i:i+1])
                    self.cache_size = new_max_capacity_prompts
                    if self.layer_idx == 0:
                        print("self.cache_size",self.cache_size)
                    
                    if "have_a_try" in method_name:
                        max_capacity_prompt = self.max_capacity_prompt
                        if q_len < max_capacity_prompt:
                            max_capacity_prompt = q_len
                        attn_weights = attn_weights_now
                        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                        if self.pooling == 'avgpool':
                            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        elif self.pooling == 'maxpool':
                            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        else:
                            raise ValueError('Pooling method not supported')

                        group_num_layer = manager.layer_groups[self.layer_idx]
                        capacity = max_capacity_prompt*3//2
                        
                        num = num_attention_heads // group_num_layer
                        group_nums = [num for _ in range(group_num_layer)]
                        if num*group_num_layer < num_attention_heads:
                            for i in range(num_attention_heads-num*group_num_layer):
                                group_nums[i] += 1
                        group_idx = list(accumulate(group_nums))
                        group_idx = [0] + group_idx
                        min_num = 128
                        max_num = 640
                        if group_num_layer != 1:
                            steps = (max_num - min_num) // (group_num_layer-1)
                                        
                            max_capacity_prompts = []
                            self.recent_indices_generate = []
                            for i in range(group_num_layer):
                                max_capacity_prompts.append(max_num - steps*(group_num_layer-i-1))
                                self.recent_indices_generate.append(torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,group_nums[i],-1))
                            
                            for i in range(group_num_layer):
                                max_capacity_prompts.append(max_num - steps*(31-i))
                            
                            max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                        else:
                            self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_attention_heads,-1)]
                            max_capacity_prompts = [capacity//2]
                            
                            
                        revise_key_states = []
                        revise_value_states = []
                        self.head_pattern = []
                        self.attn_weights = []
                        
                        new_max_capacity_prompts = []
                        for i in range(group_num_layer-1,-1,-1):
                            max_capacity_prompt = max_capacity_prompts[i]
                            new_max_capacity_prompts.append(max_capacity_prompt)
                            top_k = max_capacity_prompt - self.window_size
                            indices = attn_cache.topk(top_k, dim=-1).indices
                            indices1 = indices.sort(dim=-1).values
                            recent_indices = self.recent_indices_generate[i]+q_len
                            indices_1 = torch.cat([indices1[:,self.head_indices1[group_idx[i]:group_idx[i+1]],:],recent_indices],dim=-1)
                            indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                            indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                            revise_key_states_single=key_states[:,self.head_indices1[group_idx[i]:group_idx[i+1]],:,:].gather(dim = 2, index = indices_expanded)
                            revise_value_states_single=value_states[:,self.head_indices1[group_idx[i]:group_idx[i+1]],:,:].gather(dim = 2, index = indices_expanded)
                            attn_weights_1=attn_weights[:, self.head_indices1[group_idx[i]:group_idx[i+1]], :,:].gather(dim = -1, index = indices_attn)

                            revise_key_states.append(revise_key_states_single)
                            revise_value_states.append(revise_value_states_single)
                            self.attn_weights.append(attn_weights_1)
                            self.head_pattern.append(self.head_indices1[group_idx[i]:group_idx[i+1]])
                        self.cache_size = new_max_capacity_prompts
                        if self.layer_idx == 0:
                            print("self.cache_size layer0",self.cache_size)
                        if self.layer_idx == 31:
                            print("self.cache_size layer31",self.cache_size)
                    
                    if "stream_head" in method_name:
                        if q_len < max_capacity_prompt:
                            max_capacity_prompt = q_len
                        attn_weights = attn_weights_now
                        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                        if self.pooling == 'avgpool':
                            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        elif self.pooling == 'maxpool':
                            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        else:
                            raise ValueError('Pooling method not supported')
                        group_num_layer = 2
                        stream_num = 31
                        retrieval_num = num_attention_heads-stream_num
                        self.recent_indices_generate = []
                        self.recent_indices_generate.append(torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,stream_num,-1))
                        self.recent_indices_generate.append(torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,retrieval_num,-1))
                        
                        group_idx = [0,stream_num,32]
                        max_capacity_prompts = [20,q_len]
                        
                        revise_key_states = []
                        revise_value_states = []
                        self.head_pattern = []
                        self.attn_weights = []
                        new_max_capacity_prompts = []
                        for i in range(group_num_layer):
                            max_capacity_prompt = max_capacity_prompts[i]
                            new_max_capacity_prompts.append(max_capacity_prompt)
                            top_k = max_capacity_prompt - self.window_size
                            indices = attn_cache.topk(top_k, dim=-1).indices
                            indices1 = indices.sort(dim=-1).values
                            recent_indices = self.recent_indices_generate[i]+q_len
                            indices_1 = torch.cat([indices1[:,self.head_indices1[group_idx[i]:group_idx[i+1]],:],recent_indices],dim=-1)
                            indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                            indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                            revise_key_states_single=key_states[:,self.head_indices1[group_idx[i]:group_idx[i+1]],:,:].gather(dim = 2, index = indices_expanded)
                            revise_value_states_single=value_states[:,self.head_indices1[group_idx[i]:group_idx[i+1]],:,:].gather(dim = 2, index = indices_expanded)
                            attn_weights_1=attn_weights[:, self.head_indices1[group_idx[i]:group_idx[i+1]], :,:].gather(dim = -1, index = indices_attn)

                            revise_key_states.append(revise_key_states_single)
                            revise_value_states.append(revise_value_states_single)
                            self.attn_weights.append(attn_weights_1)
                            self.head_pattern.append(self.head_indices1[group_idx[i]:group_idx[i+1]])
                        self.cache_size = new_max_capacity_prompts
                        if self.layer_idx == 0:
                            print("self.cache_size layer0",self.cache_size)
                        if self.layer_idx == 31:
                            print("self.cache_size layer31",self.cache_size)
                    
                    if "s2_attention" in method_name and "new" not in method_name:
                        block_size = 8
                        sum_total = q_len // block_size
                        global_token_group_num = sum_total*2//3
                        local_token_group_num = sum_total*1//3
                        # print("global_token_group_num",global_token_group_num)
                        # print("local_token_group_num",local_token_group_num)
                        # assert 1==0
                        
                        head_group_num = 2
                        head_group = num_attention_heads // head_group_num
                        
                        token_len_per_group = q_len // (global_token_group_num+local_token_group_num)
                        # print("token_len_per_group",token_len_per_group)
                        head_per_group_token_group_num = global_token_group_num // head_group_num
                        head_per_group_token_sum = head_group_num*token_len_per_group
                        # print("head_per_group_token_group_num",head_per_group_token_group_num)
                        # print("head_per_group_token_sum",head_per_group_token_sum)
                        
                        revise_key_states = []
                        revise_value_states = []
                        self.head_pattern = []
                        
                        my_indices = []
                        for i in range(head_group_num):
                            tensor_sum = []
                            for j in range(head_per_group_token_group_num):
                                tensor_sum.append(torch.arange(head_per_group_token_sum*j+token_len_per_group*i,head_per_group_token_sum*j+token_len_per_group*(i+1),device='cuda'))
                            tensor1 = torch.cat(tensor_sum,dim=0)
                            tensor2 = torch.arange(token_len_per_group*global_token_group_num,q_len,device='cuda')
                            indice = torch.cat([tensor1,tensor2],dim=0)[None,:]
                            # if i ==0:
                            #     torch.set_printoptions(threshold=torch.inf)
                                # print("indice",indice)
                                # assert 1==0
                            # logger.info(f"indices.shape:{indice.shape}")
                            head_indices = torch.tensor([ i+head_group_num*j for j in range(head_group)],device='cuda')[:,None]
                            # print("indice.shape",indice.shape)
                            # assert 1==0
                            revise_key_states_single=key_states[:,head_indices,indice,:]
                            revise_value_states_single=value_states[:,head_indices,indice,:]
                            # logger.info(f"revise_key_states_single.shape:{revise_key_states_single.shape}")
                            my_indices.append(indice)
                            revise_key_states.append(revise_key_states_single)
                            revise_value_states.append(revise_value_states_single)
                            self.head_pattern.append(head_indices.reshape(-1))
                        my_indices = torch.cat(my_indices,dim=0)
                        num_tokens = torch.unique(my_indices)
                        # print("num_tokens/qlen",num_tokens.shape[0]/q_len)

                    if "new_s2_attention" in method_name:
                        block_size = 8
                        recent_size = 64
                        global_token = 320
                        
                        attn_weights = attn_weights_now_all
                        attn_weights_sum = attn_weights[:, :, :, :-recent_size].sum(dim = -2)
                        indices = attn_weights_sum.topk(global_token, dim=-1).indices
                        recent_indices = torch.arange(q_len-recent_size,q_len,device='cuda').view(1, 1, -1).expand(self.bsz,num_attention_heads,-1)
                        my_indices = torch.cat([indices,recent_indices],dim=-1)
                        indices_expanded = my_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                        
                        revise_key_states = []
                        revise_value_states = []
                        self.head_pattern = []
                        for i in range(1):
                            revise_key_states_single=key_states.gather(dim = 2, index = indices_expanded)
                            revise_value_states_single=value_states.gather(dim = 2, index = indices_expanded)
                            revise_key_states.append(revise_key_states_single)
                            revise_value_states.append(revise_value_states_single)
                            self.head_pattern.append(torch.arange(32))
                        # print("revise_key_states_single.shape",revise_key_states_single.shape)
                        num_tokens = torch.unique(torch.flatten(indices_expanded))
                        print(f"self.layer_idx:{self.layer_idx},num_tokens/qlen:{num_tokens.shape[0]/q_len}")
                    # if "comparison" in method_name and "reranking" not in method_name:
                    #     recent_size = 200
                    #     max_capacity_prompt = self.max_capacity_prompt
                    #     block_size = 16
                    #     block_num = (q_len-recent_size) // block_size
                    #     if block_num * block_size < q_len:
                    #         retain_tokens = q_len-recent_size - block_num * block_size
                    #     else:
                    #         retain_tokens = 0
                    #     # 规整化
                    #     attn_weights = attn_weights_now
                    #     attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                    #     if self.pooling == 'avgpool':
                    #         attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    #     elif self.pooling == 'maxpool':
                    #         attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    #     else:
                    #         raise ValueError('Pooling method not supported')
                        
                    #     # 选取topk
                    #     # print("attn_cache.shape",attn_cache.shape)
                    #     # print("q_len-retain_tokens",q_len-retain_tokens)
                    #     indices = attn_cache.topk(q_len-recent_size-retain_tokens, dim=-1).indices.sort(dim=-1).values
                    #     indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    #     new_query = query_states.gather(dim=2, index=indices_expanded).reshape(self.bsz, num_attention_heads, -1, block_size, head_dim)
                        
                    #     # 进行筛选 bsz,head,group_num,block_size,head_dim
                    #         #计算矩阵熵
                    #     entropys = torch.from_numpy(get_entropy_head_5_svd_dimensions_single_batch(new_query,32))
                    #         #选择topk
                    #     select_indices = entropys.topk(80, dim=-1).indices.sort(dim=-1).values
                    #     torch.set_printoptions(threshold=torch.inf)
                    #     # print("select_indices",select_indices)
                    #     def expand_row(row):
                    #         nonlocal block_size
                    #         expanded = torch.cat([torch.arange(x * block_size, (x + 1) * block_size,device="cuda") for x in row])
                    #         return expanded
                        
                    #     output_tensor = torch.stack([expand_row(row) for row in select_indices]).unsqueeze(0)
                    #     # print("torch.arange(q_len-self.window_size,q_len,device='cuda').unsqueeze(0).unsqueeze(0).expand(self.bsz,num_attention_heads,-1)",torch.arange(q_len-self.window_size,q_len,device='cuda').unsqueeze(0).unsqueeze(0).expand(self.bsz,num_attention_heads,-1))
                    #     output_tensor = indices.gather(dim = 2, index = output_tensor)
                    #     output_tensor = torch.cat([output_tensor,torch.arange(q_len-self.window_size,q_len,device='cuda').unsqueeze(0).unsqueeze(0).expand(self.bsz,num_attention_heads,-1)],dim=-1)
                        
                    #     # print("output_tensor",output_tensor[0,:,:])
                    #     output_tensor_expand = output_tensor.reshape(self.bsz, num_attention_heads, output_tensor.shape[-1], -1).expand(-1, -1, -1, head_dim)
                        
                    #     revise_key_states = []
                    #     revise_value_states = []
                    #     self.head_pattern = []
                    #     # print("output_tensor.shape",output_tensor.shape)
                    #     # assert 1==0
                    #     for i in range(1):
                    #         revise_key_states_single=key_states.gather(dim = 2, index = output_tensor_expand)
                    #         revise_value_states_single=value_states.gather(dim = 2, index = output_tensor_expand)
                            
                    #         revise_key_states.append(revise_key_states_single)
                    #         revise_value_states.append(revise_value_states_single)
                    #         self.head_pattern.append(torch.arange(32))
                    #     num_tokens = torch.unique(torch.flatten(output_tensor))
                    #     print(f"self.layer_idx:{self.layer_idx},num_tokens/qlen:{num_tokens.shape[0]/q_len}")
                    
                    # if "reranking_comparison" in method_name and "new" not in method_name:
                    #     recent_size = self.window_size
                    #     max_capacity_prompt = self.max_capacity_prompt
                    #     block_size = 16
                    #     block_num = (q_len-recent_size) // block_size
                    #     if block_num * block_size < q_len:
                    #         retain_tokens = q_len-recent_size - block_num * block_size
                    #     else:
                    #         retain_tokens = 0
                    #     # 规整化
                    #     attn_weights = attn_weights_now
                    #     attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                    #     if self.pooling == 'avgpool':
                    #         attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    #     elif self.pooling == 'maxpool':
                    #         attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    #     else:
                    #         raise ValueError('Pooling method not supported')
                        
                    #     def expand_row(row):
                    #         nonlocal block_size
                    #         expanded = torch.cat([torch.arange(x * block_size, (x + 1) * block_size,device="cuda") for x in row])
                    #         return expanded
                        
                    #     # 选取topk
                    #     # print("attn_cache.shape",attn_cache.shape)
                    #     # print("q_len-retain_tokens",q_len-retain_tokens)
                    #     indices = attn_cache.topk(q_len-recent_size-retain_tokens, dim=-1).indices.sort(dim=-1).values
                    #     attn_cache_new = attn_cache.gather(dim = -1, index = indices).reshape(self.bsz, num_attention_heads, -1, block_size)
                    #         #初筛160组
                    #         #选出每组最大值
                    #     indices_group = attn_cache_new.topk(1, dim=-1).indices.sort(dim=-1).values
                    #         #选出最大值最大的160组下标
                    #     attn_cache_new = attn_cache_new.gather(dim = -1, index = indices_group)
                    #     indices_group = attn_cache_new.topk(160, dim=-2).indices.sort(dim=-2).values.squeeze(-1)
                    #         #选出具体的值
                    #     output_tensor1 = torch.stack([expand_row(row) for row in indices_group.squeeze(0)]).unsqueeze(0)
                    #     output_tensor1 = indices.gather(dim = 2, index = output_tensor1)
                    #     indices_expanded = output_tensor1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    #     new_query = query_states.gather(dim=2, index=indices_expanded).reshape(self.bsz, num_attention_heads, -1, block_size, head_dim)
                    #     # entropys = torch.from_numpy(get_entropy_head_5_svd_dimensions_single_batch(new_query,32))
                    #         #矩阵熵reranking
                    #     select_indices = entropys.topk(80, dim=-1).indices.sort(dim=-1).values
                    #         #筛选出剩下的
                    #     output_tensor = torch.stack([expand_row(row) for row in select_indices]).unsqueeze(0)
                    #     output_tensor = output_tensor1.gather(dim = 2, index = output_tensor)
                    #     output_tensor = torch.cat([output_tensor,torch.arange(q_len-self.window_size,q_len,device='cuda').unsqueeze(0).unsqueeze(0).expand(self.bsz,num_attention_heads,-1)],dim=-1)
                        
                    #     # print("output_tensor",output_tensor[0,:,:])
                    #     output_tensor_expand = output_tensor.reshape(self.bsz, num_attention_heads, output_tensor.shape[-1], -1).expand(-1, -1, -1, head_dim)
                        
                    #     revise_key_states = []
                    #     revise_value_states = []
                    #     self.head_pattern = []
                    #     # torch.set_printoptions(threshold=torch.inf)
                    #     # print("output_tensor",output_tensor)
                    #     # print("output_tensor.shape",output_tensor.shape)
                    #     # assert 1==0
                    #     for i in range(1):
                    #         revise_key_states_single=key_states.gather(dim = 2, index = output_tensor_expand)
                    #         revise_value_states_single=value_states.gather(dim = 2, index = output_tensor_expand)
                            
                    #         revise_key_states.append(revise_key_states_single)
                    #         revise_value_states.append(revise_value_states_single)
                    #         self.head_pattern.append(torch.arange(32))
                    #     num_tokens = torch.unique(torch.flatten(output_tensor))
                    #     print(f"self.layer_idx:{self.layer_idx},num_tokens/qlen:{num_tokens.shape[0]/q_len}")
                    
                    if "new_reranking_comparison" in method_name :
                        recent_size = self.window_size
                        max_capacity_prompt = self.max_capacity_prompt
                        block_size = 8
                        block_num = (q_len-recent_size) // block_size
                        if block_num * block_size < q_len:
                            retain_tokens = q_len-recent_size - block_num * block_size
                        else:
                            retain_tokens = 0
                        # 规整化
                        attn_weights = attn_weights_now
                        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                        if self.pooling == 'avgpool':
                            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        elif self.pooling == 'maxpool':
                            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        else:
                            raise ValueError('Pooling method not supported')
                        
                        def expand_row(row):
                            nonlocal block_size
                            expanded = torch.cat([torch.arange(x * block_size, (x + 1) * block_size,device="cuda") for x in row])
                            return expanded
                        
                        # 选取topk
                        # print("attn_cache.shape",attn_cache.shape)
                        # print("q_len-retain_tokens",q_len-retain_tokens)
                        indices = attn_cache.topk(q_len-recent_size-retain_tokens, dim=-1).indices.sort(dim=-1).values
                        attn_cache_new = attn_cache.gather(dim = -1, index = indices).reshape(self.bsz, num_attention_heads, -1, block_size)
                            #初筛160组
                            #选出每组最大值
                        indices_group = attn_cache_new.topk(1, dim=-1).indices.sort(dim=-1).values
                            #选出最大值最大的160组下标
                        attn_cache_new = attn_cache_new.gather(dim = -1, index = indices_group)
                        indices_group = attn_cache_new.topk(376//block_size, dim=-2).indices.sort(dim=-2).values.squeeze(-1)
                            #选出具体的值
                        output_tensor1 = torch.stack([expand_row(row) for row in indices_group.squeeze(0)]).unsqueeze(0)
                        output_tensor1 = indices.gather(dim = 2, index = output_tensor1)
                        
                        recent_indices = torch.arange(q_len-recent_size,q_len,device='cuda').view(1, 1, -1).expand(self.bsz,num_attention_heads,-1)
                        
                        # output_tensor1 = attn_cache.topk(376, dim=-1).indices.sort(dim=-1).values
                        output_tensor1 = torch.cat([output_tensor1,recent_indices],dim=-1)
                        indices_expanded = output_tensor1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                        # new_query = query_states.gather(dim=2, index=indices_expanded).reshape(self.bsz, num_attention_heads, -1, block_size, head_dim)
                        # entropys = torch.from_numpy(get_entropy_head_5_svd_dimensions_single_batch(new_query,32))
                        #     #矩阵熵reranking
                        # select_indices = entropys.topk(376//block_size, dim=-1).indices.sort(dim=-1).values
                        #     #筛选出剩下的
                        # output_tensor = torch.stack([expand_row(row) for row in select_indices]).unsqueeze(0)
                        # output_tensor = output_tensor1.gather(dim = 2, index = output_tensor)
                        # output_tensor = torch.cat([output_tensor,torch.arange(q_len-self.window_size,q_len,device='cuda').unsqueeze(0).unsqueeze(0).expand(self.bsz,num_attention_heads,-1)],dim=-1)
                        
                        # print("output_tensor",output_tensor[0,:,:])
                        # output_tensor_expand = output_tensor.reshape(self.bsz, num_attention_heads, output_tensor.shape[-1], -1).expand(-1, -1, -1, head_dim)
                        
                        revise_key_states = []
                        revise_value_states = []
                        self.head_pattern = []
                        # torch.set_printoptions(threshold=torch.inf)
                        # print("output_tensor",output_tensor)
                        # print("output_tensor.shape",output_tensor.shape)
                        # assert 1==0
                        for i in range(1):
                            revise_key_states_single=key_states.gather(dim = 2, index = indices_expanded)
                            revise_value_states_single=value_states.gather(dim = 2, index = indices_expanded)
                            
                            revise_key_states.append(revise_key_states_single)
                            revise_value_states.append(revise_value_states_single)
                            self.head_pattern.append(torch.arange(32))
                        num_tokens = torch.unique(torch.flatten(indices_expanded))
                        print(f"self.layer_idx:{self.layer_idx},num_tokens/qlen:{num_tokens.shape[0]/q_len}")

            elif method_name in manager.chai:
                revise_key_states = key_states
                revise_value_states = value_states
            elif "pyramidkv_generate" in method_name:
                self.beta = 20
                my_max = int(self.max_capacity_prompt * 1.5)
                min_num = self.max_capacity_prompt // self.beta
                max_num = my_max - min_num
                if max_num >= q_len:
                    max_num = q_len
                    min_num = my_max - max_num
                steps = (max_num - min_num) // self.num_hidden_layers 
                max_capacity_prompt = min_num + (num_hidden_layers-self.layer_idx) * steps
                if q_len < max_capacity_prompt:
                    max_capacity_prompt = q_len
                if max_capacity_prompt < self.window_size:
                    max_capacity_prompt = self.window_size
                self.cache_size = max_capacity_prompt
                attn_weights = attn_weights_now
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
                revise_key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                revise_value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                attn_weights_cur= attn_weights[:,:,-self.window_size:,-self.window_size:]
                self.attn_weights = torch.cat([attn_weights_compress,attn_weights_cur],dim=-1)
                if revise_key_states.shape != (bsz, num_heads, max_capacity_prompt, head_dim):
                    print("revise_key_states.shape",revise_key_states.shape)
                    print("max_capacity_prompt",max_capacity_prompt)
                    raise ValueError("revise_key_states.shape != (bsz, num_heads, max_capacity_prompt, head_dim)")
            elif method_name in manager.pyramidkv_uncomp and "group" not in method_name:
                self.beta = 20
                my_max = int(self.max_capacity_prompt * 1.5)
                min_num = self.max_capacity_prompt // self.beta
                max_num = my_max - min_num
                if max_num >= q_len:
                    max_num = q_len
                    min_num = my_max - max_num
                steps = (max_num - min_num) // self.num_hidden_layers 
                max_capacity_prompt = min_num + (num_hidden_layers-self.layer_idx) * steps
                if q_len < max_capacity_prompt:
                    max_capacity_prompt = q_len
                if max_capacity_prompt < self.window_size:
                    max_capacity_prompt = self.window_size
                # 分配到当前层的max_capacity_prompt
                
                if self.window_size > max_capacity_prompt//2:
                    self.window_size = max_capacity_prompt//2
                    self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_attention_heads//2,-1),torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers//2,-1)]
                
                # max_capacity_prompts = [max_capacity_prompt//2,max_capacity_prompt]
                
                capacity = max_capacity_prompt*3//2
                steps = capacity // 6
                max_capacity_prompts = [steps,capacity-steps]
                
                print(f"self.layer_idx:{self.layer_idx} , max_capacity_prompts:{max_capacity_prompts}")
                attn_weights = attn_weights_now
                attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')

                top_k = max_capacity_prompts[1] - self.window_size
                indices = attn_cache.topk(top_k, dim=-1).indices
                indices1 = indices.sort(dim=-1).values
                for i in range(len(self.recent_indices_generate)):
                    self.recent_indices_generate[i] = self.recent_indices_generate[i].to(indices1.device)
                self.head_indices1 = self.head_indices1.to(indices1.device)
                recent_indices = self.recent_indices_generate[0]+q_len
                num_heads = num_attention_heads // 2
                indices_1 = torch.cat([indices1[:,self.head_indices1[-num_heads:],:],recent_indices],dim=-1)
                indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                revise_key_states=key_states[:,self.head_indices1[-num_heads:],:,:].gather(dim = 2, index = indices_expanded)
                revise_value_states=value_states[:,self.head_indices1[-num_heads:],:,:].gather(dim = 2, index = indices_expanded)
                self.attn_weights_1=attn_weights[:, self.head_indices1[-num_heads:], :,:].gather(dim = -1, index = indices_attn)

                max_capacity_prompt_2 = max_capacity_prompts[0]
                top_k2 = max_capacity_prompt_2 - self.window_size
                indices = attn_cache.topk(top_k2, dim=-1).indices
                indices_2 = indices.sort(dim=-1).values
                indices_2 = torch.cat([indices_2[:,self.head_indices1[:num_heads],:],recent_indices],dim=-1)
                indices_expanded_2  = indices_2.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices_attn_2 = indices_2.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                revise_value_states_2 = value_states[:,self.head_indices1[:num_heads],:,:].gather(dim=2,index=indices_expanded_2)
                revise_key_states_2 = key_states[:,self.head_indices1[:num_heads],:,:].gather(dim=2, index=indices_expanded_2)
                self.attn_weights_2 = attn_weights[:, self.head_indices1[:num_heads], :,:].gather(dim = -1, index = indices_attn_2)
                key1 = revise_key_states
                key2 = revise_key_states_2
                value1 = revise_value_states
                value2 = revise_value_states_2
                revise_key_states = [key1, key2]
                revise_value_states = [value1, value2]
                self.head_pattern = [self.head_indices1[-num_heads:], self.head_indices1[:num_heads]]
                self.attn_weights = [self.attn_weights_1, self.attn_weights_2]
                self.cache_size = [max_capacity_prompts[1], max_capacity_prompts[0]]   
            elif method_name in manager.pyramidkv_uncomp and "group" in method_name:
                self.beta = 20
                my_max = int(self.max_capacity_prompt * 1.5)
                min_num = self.max_capacity_prompt // self.beta
                max_num = my_max - min_num
                if max_num >= q_len:
                    max_num = q_len
                    min_num = my_max - max_num
                steps = (max_num - min_num) // self.num_hidden_layers 
                max_capacity_prompt = min_num + (num_hidden_layers-self.layer_idx) * steps
                if q_len < max_capacity_prompt:
                    max_capacity_prompt = q_len
                if max_capacity_prompt < self.window_size:
                    max_capacity_prompt = self.window_size
                # 分配到当前层的max_capacity_prompt
                if self.window_size > max_capacity_prompt//2:
                    self.window_size = max_capacity_prompt//2
                    self.recent_indices_generate = [torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_attention_heads//2,-1),torch.arange(-self.window_size,0,device='cuda').view(1, 1, -1).expand(self.bsz,self.num_hidden_layers//2,-1)]
                
                # max_capacity_prompt = self.max_capacity_prompt
                if q_len < max_capacity_prompt:
                    max_capacity_prompt = q_len
                attn_weights = attn_weights_now
                attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')
                
                if "group3" in method_name and "group32" not in method_name:
                    max_capacity_prompts = [max_capacity_prompt,max_capacity_prompt*3//4,max_capacity_prompt//2]

                    max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                    # if self.layer_idx == 0:
                    print(f"self.layer_idx:{self.layer_idx} , max_capacity_prompts:{max_capacity_prompts}")
                    max_capacity_prompt = max_capacity_prompts[0]
                    top_k = max_capacity_prompt - self.window_size
                    indices = attn_cache.topk(top_k, dim=-1).indices
                    indices1 = indices.sort(dim=-1).values
                    recent_indices = self.recent_indices_generate[0]+q_len
                    indices_1 = torch.cat([indices1[:,self.head_indices1[-11:],:],recent_indices],dim=-1)
                    indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_key_states=key_states[:,self.head_indices1[-11:],:,:].gather(dim = 2, index = indices_expanded)
                    revise_value_states=value_states[:,self.head_indices1[-11:],:,:].gather(dim = 2, index = indices_expanded)
                    attn_weights_1=attn_weights[:, self.head_indices1[-11:], :,:].gather(dim = -1, index = indices_attn)

                    max_capacity_prompt = max_capacity_prompts[1]
                    top_k2 = max_capacity_prompt - self.window_size
                    indices_2 = indices[:,:,:top_k2].sort(dim=-1).values
                    recent_indices_1 = self.recent_indices_generate[1]+q_len
                    indices_2 = torch.cat([indices_2[:,self.head_indices1[11:-11],:],recent_indices_1],dim=-1)
                    indices_expanded_2  = indices_2.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_2 = indices_2.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_2 = value_states[:,self.head_indices1[11:-11],:,:].gather(dim=2,index=indices_expanded_2)
                    revise_key_states_2 = key_states[:,self.head_indices1[11:-11],:,:].gather(dim=2, index=indices_expanded_2)
                    attn_weights_2 = attn_weights[:, self.head_indices1[11:-11], :,:].gather(dim = -1, index = indices_attn_2)
                    
                    max_capacity_prompt = max_capacity_prompts[2]
                    top_k3 = max_capacity_prompt - self.window_size
                    indices_3 = indices[:,:,:top_k3].sort(dim=-1).values
                    indices_3 = torch.cat([indices_3[:,self.head_indices1[:11],:],recent_indices],dim=-1)
                    indices_expanded_3  = indices_3.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_3 = indices_3.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_3 = value_states[:,self.head_indices1[:11],:,:].gather(dim=2,index=indices_expanded_3)
                    revise_key_states_3 = key_states[:,self.head_indices1[:11],:,:].gather(dim=2, index=indices_expanded_3)
                    attn_weights_3 = attn_weights[:, self.head_indices1[:11], :,:].gather(dim = -1, index = indices_attn_3)
                    
                    key1 = revise_key_states
                    key2 = revise_key_states_2
                    key3 = revise_key_states_3
                    value1 = revise_value_states
                    value2 = revise_value_states_2
                    value3 = revise_value_states_3
                    revise_key_states = [key1, key2, key3]
                    revise_value_states = [value1, value2, value3]
                    self.head_pattern = [self.head_indices1[-11:], self.head_indices1[11:-11], self.head_indices1[:11]]
                    self.attn_weights = [attn_weights_1, attn_weights_2, attn_weights_3]
                    self.cache_size = max_capacity_prompts
                elif "group4" in method_name:
                    capacity = max_capacity_prompt*3//2
                    # steps = capacity // 4
                    # if steps >= 8:
                    #     max_capacity_prompts = [steps,steps*2,steps*3,steps*4]
                    # else:
                    #     steps = 8
                    #     max_capacity_prompts = [steps,steps*2,capacity-steps*2,capacity-steps]
                    
                    steps = capacity // 18
                    if steps*3 >= 8:
                        max_capacity_prompts = [steps*3,steps*7,capacity-steps*7,capacity-steps*3][::-1]
                    else:
                        steps = 9
                        max_capacity_prompts = [steps,steps*2,capacity-steps*2,capacity-steps][::-1]
                    
                    max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                    
                    if q_len < max_capacity_prompt:
                        max_capacity_prompt = q_len
                    attn_weights = attn_weights_now
                    attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim = -2)
                    
                    if self.pooling == 'avgpool':
                        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    elif self.pooling == 'maxpool':
                        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    else:
                        raise ValueError('Pooling method not supported')
                    
                    # if self.max_capacity_prompt == 86:
                    #     max_capacity_prompts = [112,80,48,16]
                    #     if "new1" in method_name:
                    #         max_capacity_prompts = [116,81,47,12]
                    # elif self.max_capacity_prompt == 512:
                    #     max_capacity_prompts = [512,427,341,256]
                    # else:
                    #     max_capacity_prompts = []
                    #     max_cap = max_capacity_prompt
                    #     min_cap = max_capacity_prompt // 2
                    #     allowance = (max_cap-min_cap) // 3
                    #     for i in range(4):
                    #         max_capacity_prompts.append(max_cap - allowance*i)
                    # max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                    
                    if self.layer_idx == 0:
                        print("max_capacity_prompts",max_capacity_prompts)
                    max_capacity_prompt = max_capacity_prompts[0]
                    top_k = max_capacity_prompt - self.window_size
                    indices = attn_cache.topk(top_k, dim=-1).indices
                    indices1 = indices.sort(dim=-1).values
                    recent_indices = self.recent_indices_generate[0]+q_len
                    indices_1 = torch.cat([indices1[:,self.head_indices1[-8:],:],recent_indices],dim=-1)
                    indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_key_states=key_states[:,self.head_indices1[-8:],:,:].gather(dim = 2, index = indices_expanded)
                    revise_value_states=value_states[:,self.head_indices1[-8:],:,:].gather(dim = 2, index = indices_expanded)
                    attn_weights_1=attn_weights[:, self.head_indices1[-8:], :,:].gather(dim = -1, index = indices_attn)

                    max_capacity_prompt = max_capacity_prompts[1]
                    top_k2 = max_capacity_prompt - self.window_size
                    indices_2 = indices[:,:,:top_k2].sort(dim=-1).values
                    recent_indices = self.recent_indices_generate[1]+q_len
                    recent_indices_1 = self.recent_indices_generate[1]+q_len
                    indices_2 = torch.cat([indices_2[:,self.head_indices1[16:-8],:],recent_indices_1],dim=-1)
                    indices_expanded_2  = indices_2.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_2 = indices_2.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_2 = value_states[:,self.head_indices1[16:-8],:,:].gather(dim=2,index=indices_expanded_2)
                    revise_key_states_2 = key_states[:,self.head_indices1[16:-8],:,:].gather(dim=2, index=indices_expanded_2)
                    attn_weights_2 = attn_weights[:, self.head_indices1[16:-8], :,:].gather(dim = -1, index = indices_attn_2)
                    
                    max_capacity_prompt = max_capacity_prompts[2]
                    top_k3 = max_capacity_prompt - self.window_size
                    recent_indices = self.recent_indices_generate[2]+q_len
                    indices_3 = indices[:,:,:top_k3].sort(dim=-1).values
                    indices_3 = torch.cat([indices_3[:,self.head_indices1[8:16],:],recent_indices],dim=-1)
                    indices_expanded_3  = indices_3.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_3 = indices_3.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_3 = value_states[:,self.head_indices1[8:16],:,:].gather(dim=2,index=indices_expanded_3)
                    revise_key_states_3 = key_states[:,self.head_indices1[8:16],:,:].gather(dim=2, index=indices_expanded_3)
                    attn_weights_3 = attn_weights[:, self.head_indices1[8:16], :,:].gather(dim = -1, index = indices_attn_3)
                    
                    max_capacity_prompt = max_capacity_prompts[3]
                    top_k4 = max_capacity_prompt - self.window_size
                    indices_4 = indices[:,:,:top_k4].sort(dim=-1).values
                    recent_indices = self.recent_indices_generate[3]+q_len
                    indices_4 = torch.cat([indices_4[:,self.head_indices1[:8],:],recent_indices],dim=-1)
                    indices_expanded_4  = indices_4.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn_4 = indices_4.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    revise_value_states_4 = value_states[:,self.head_indices1[:8],:,:].gather(dim=2,index=indices_expanded_4)
                    revise_key_states_4 = key_states[:,self.head_indices1[:8],:,:].gather(dim=2, index=indices_expanded_4)
                    attn_weights_4 = attn_weights[:, self.head_indices1[:8], :,:].gather(dim = -1, index = indices_attn_4)
                    
                    key1,key2,key3,key4 = revise_key_states,revise_key_states_2,revise_key_states_3,revise_key_states_4
                    value1,value2,value3,value4 = revise_value_states,revise_value_states_2,revise_value_states_3,revise_value_states_4
                    revise_key_states = [key1, key2, key3, key4]
                    revise_value_states = [value1, value2, value3, value4]
                    self.head_pattern = [self.head_indices1[-8:], self.head_indices1[16:-8], self.head_indices1[8:16], self.head_indices1[:8]]
                    self.attn_weights = [attn_weights_1, attn_weights_2, attn_weights_3, attn_weights_4]
                    self.cache_size = max_capacity_prompts  
                elif "group8" in method_name:
                    # print("max_capacity_prompt",max_capacity_prompt)
                    capacity = max_capacity_prompt*3//2
                    steps = capacity // 42
                    if steps*7 >= 8:
                        max_capacity_prompts = [steps*7,steps*11,steps*15,steps*19,capacity-steps*19,capacity-steps*15,capacity-steps*11,capacity-steps*7]
                    else:
                        steps = 9
                        max_capacity_prompts = [steps,steps,steps,steps,capacity-steps,capacity-steps,capacity-steps,capacity-steps]
                    
                    max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                    revise_key_states = []
                    revise_value_states = []
                    self.head_pattern = []
                    self.attn_weights = []
                    if self.layer_idx == 0:
                        print("max_capacity_prompts",max_capacity_prompts)
                    new_max_capacity_prompts = []
                    for i in range(7,-1,-1):
                        max_capacity_prompt = max_capacity_prompts[i]
                        new_max_capacity_prompts.append(max_capacity_prompt)
                        top_k = max_capacity_prompt - self.window_size
                        indices = attn_cache.topk(top_k, dim=-1).indices
                        indices1 = indices.sort(dim=-1).values
                        recent_indices = self.recent_indices_generate[i]+q_len
                        indices_1 = torch.cat([indices1[:,self.head_indices1[4*i:4*(i+1)],:],recent_indices],dim=-1)
                        indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                        indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                        revise_key_states_single=key_states[:,self.head_indices1[4*i:4*(i+1)],:,:].gather(dim = 2, index = indices_expanded)
                        revise_value_states_single=value_states[:,self.head_indices1[4*i:4*(i+1)],:,:].gather(dim = 2, index = indices_expanded)
                        attn_weights_1=attn_weights[:, self.head_indices1[4*i:4*(i+1)], :,:].gather(dim = -1, index = indices_attn)

                        revise_key_states.append(revise_key_states_single)
                        revise_value_states.append(revise_value_states_single)
                        self.attn_weights.append(attn_weights_1)
                        self.head_pattern.append(self.head_indices1[4*i:4*(i+1)])
                    self.cache_size = new_max_capacity_prompts
                elif "group32" in method_name:
                    capacity = max_capacity_prompt*3//2
                    # steps = capacity // 33
                    max_capacity_prompts = []
                    # if steps < self.window_size:
                    #     first = self.window_size + 2
                    #     end = capacity-first
                    #     steps = (end-first)//31
                    
                    #     for i in range (32):
                    #         max_capacity_prompts.append(int(first+steps*i))
                    # else:
                    #     for i in range (32):
                    #         max_capacity_prompts.append(int(steps*(i+1)))
                    
                    steps = capacity/186
                    min_num = int(steps*31)
                    max_num = capacity-min_num
                    for i in range(32):
                        max_capacity_prompts.append(int(max_num - steps*4*(31-i)))
                    
                    max_capacity_prompts = [min(prompt, q_len) for prompt in max_capacity_prompts]
                    
                    revise_key_states = []
                    revise_value_states = []
                    self.head_pattern = []
                    self.attn_weights = []
                    
                    new_max_capacity_prompts = []
                    for i in range(31,-1,-1):
                        max_capacity_prompt = max_capacity_prompts[i]
                        new_max_capacity_prompts.append(max_capacity_prompt)
                        top_k = max_capacity_prompt - self.window_size
                        indices = attn_cache.topk(top_k, dim=-1).indices
                        indices1 = indices.sort(dim=-1).values
                        recent_indices = self.recent_indices_generate[i]+q_len
                        indices_1 = torch.cat([indices1[:,self.head_indices1[i:i+1],:],recent_indices],dim=-1)
                        indices_expanded  = indices_1.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                        indices_attn = indices_1.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                        revise_key_states_single=key_states[:,self.head_indices1[i:i+1],:,:].gather(dim = 2, index = indices_expanded)
                        revise_value_states_single=value_states[:,self.head_indices1[i:i+1],:,:].gather(dim = 2, index = indices_expanded)
                        attn_weights_1=attn_weights[:, self.head_indices1[i:i+1], :,:].gather(dim = -1, index = indices_attn)

                        revise_key_states.append(revise_key_states_single)
                        revise_value_states.append(revise_value_states_single)
                        self.attn_weights.append(attn_weights_1)
                        self.head_pattern.append(self.head_indices1[i:i+1])
                    self.cache_size = new_max_capacity_prompts
                    if self.layer_idx == 31:
                        print("self.cache_size",self.cache_size)
                    # print(self.cache_size)  

            if "snapkv" in method_name:
                max_capacity_prompt = self.max_capacity_prompt
                self.window_size = 50
                if q_len < max_capacity_prompt:
                    max_capacity_prompt = q_len
                self.cache_size = max_capacity_prompt
                attn_weights = attn_weights_now
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
                revise_key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                revise_value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                attn_weights_cur= attn_weights[:,:,-self.window_size:,-self.window_size:]
                self.attn_weights = torch.cat([attn_weights_compress,attn_weights_cur],dim=-1)
                if revise_key_states.shape != (bsz, num_heads, max_capacity_prompt, head_dim):
                    print("revise_key_states.shape",revise_key_states.shape)
                    print("max_capacity_prompt",max_capacity_prompt)
                    raise ValueError("revise_key_states.shape != (bsz, num_heads, max_capacity_prompt, head_dim)")
            elif "h2o" in method_name:
                attn_weights = attn_weights_now_all
                recent_size = self.recent_size = self.max_capacity_prompt*508//512
                hh_size = self.hh_size = self.max_capacity_prompt*4//512
                
                if hh_size==0:
                    hh_size = self.hh_size = 1
                    recent_size = self.recent_size = self.max_capacity_prompt - hh_size
                if q_len <= recent_size+hh_size:
                    recent_size = self.recent_size = q_len - hh_size
                self.cache_size = recent_size + hh_size
                self.attn_weights_sum = attn_weights.sum(0).sum(1)
                select_hh_scores = self.attn_weights_sum[:, :q_len - recent_size]
                _, keep_topk = torch.topk(select_hh_scores, hh_size, dim=-1)
                keep_topk = keep_topk.sort().values 
                keep_recent = torch.arange(q_len - recent_size, q_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
                keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)
                mask = torch.zeros(self.attn_weights_sum.shape, dtype=torch.bool).to(attn_weights.device)
                mask = mask.scatter(-1, keep_idx, 1)
                revise_key_states = key_states.squeeze(0)[mask].view(bsz, num_heads, -1, head_dim)
                revise_value_states = value_states.squeeze(0)[mask].view(bsz, num_heads, -1, head_dim)
                self.attn_weights_sum = self.attn_weights_sum[mask].view(num_heads, self.cache_size)
            
            return revise_key_states, revise_value_states

    def update_kv_generate(
        self,
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        new_attn_weights: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        manager = self.manager
        if manager.method_name in manager.group_sampling:
            bsz,num_heads,q_len,head_dim = key_states.shape
            attn_weights = self.attn_weights
            now_attn_weights = torch.zeros((attn_weights.shape[0],attn_weights.shape[1],attn_weights.shape[2]+1,attn_weights.shape[3]+1),device=attn_weights.device)
            now_attn_weights[:,:,:-1,:-1] = attn_weights
            now_attn_weights[:,:,:,-1] = 0
            now_attn_weights[:,:,-1:,:] = new_attn_weights
            attn_weights_sum = now_attn_weights[...,-self.window_size:,:-self.window_size].sum(-2)
            cache_size = self.cache_size
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
            
            if revise_key_states.shape != (bsz, num_heads, cache_size, head_dim):
                    print("revise_key_states.shape",revise_key_states.shape)
                    print("max_capacity_prompt",cache_size)
                    raise ValueError("revise_key_states.shape != (bsz, num_heads, max_capacity_prompt, head_dim)")
            return None
        elif "streamingllm" not in manager.method_name: # H2O
            recent_size,hh_size = self.recent_size,self.hh_size

            bsz,num_heads,q_len,head_dim = key_states.shape
            new_attn_weights = new_attn_weights.sum(0).sum(1)  
            new_attn_weights[:,:-1] += self.attn_weights_sum
            attn_weights_sum = self.attn_weights_sum = new_attn_weights
            
            select_hh_scores = self.attn_weights_sum[:, :q_len - recent_size]
            _, keep_topk = torch.topk(select_hh_scores, hh_size, dim=-1)
            
            keep_topk = keep_topk.sort().values 
            keep_recent = torch.arange(q_len - recent_size, q_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
            keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)
            mask = torch.zeros(attn_weights_sum.shape, dtype=torch.bool).to(attn_weights_sum.device)
            mask = mask.scatter_(-1, keep_idx, 1)
            past_key_value.key_cache[layer_idx] = key_states.squeeze(0)[mask].view(bsz, num_heads, -1, head_dim)
            past_key_value.value_cache[layer_idx] = value_states.squeeze(0)[mask].view(bsz, num_heads, -1, head_dim)
            self.attn_weights_sum= self.attn_weights_sum[mask].view(num_heads, self.cache_size)
        
    def update_head_kv_generate(
        self,
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        new_attn_weights_all: torch.Tensor,
        key_states_all: torch.Tensor,
        value_states_all: torch.Tensor,
        layer_idx: int,
        padding: list,
        single=None,
    ):
        method_name = self.manager.method_name
        len1,len2,len3 = len(key_states_all),len(value_states_all),len(new_attn_weights_all)
        if len1 != len2 or len2 != len3 and single is None: 
            print(f"len1:{len1} len2:{len2} len3:{len3}")
            raise ValueError("The length of key_states_all, value_states_all, new_attn_weights_all should be the same.")
        if method_name in self.manager.delet_head_set:
            attn_weights = self.attn_weights
            
            new_attn_weights,key_states,value_states = new_attn_weights_all,key_states_all,value_states_all
            bsz,num_heads,q_len,head_dim = key_states.size()
            now_attn_weights = torch.nn.functional.pad(attn_weights, (0, 1, 0, 1), mode='constant', value=0).to(torch.float32)
            now_attn_weights[:,:,-1:,:] = new_attn_weights[...,:now_attn_weights.shape[-1]] 
            attn_weights_sum = now_attn_weights[...,-self.window_size:,:-self.window_size].sum(-2)
            
            cache_size = self.cache_size
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices  = attn_cache.topk(cache_size-self.window_size, dim=-1).indices
            bsz,num_heads,q_len,_= key_states.shape
            recent_indices = self.recent_indices_generate[0]+q_len
            indices = torch.cat([indices,recent_indices],dim=-1)
            indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
            attn_weights_compress = now_attn_weights[...,-self.window_size:,:].gather(dim = -1, index = indices_attn)
            revise_key_states = key_states.gather(dim = 2, index = indices_expanded)
            revise_value_states = value_states.gather(dim = 2, index = indices_expanded)

            self.attn_weights = attn_weights_compress
            past_key_value.key_cache[layer_idx] = revise_key_states
            past_key_value.value_cache[layer_idx] = revise_value_states
            
            if revise_key_states.shape != (bsz, num_heads, cache_size, head_dim):
                print("revise_key_states.shape",revise_key_states.shape)
                print("max_capacity_prompt",cache_size)
                raise ValueError("revise_key_states.shape != (bsz, num_heads, max_capacity_prompt, head_dim)")
        elif method_name in self.manager.head_differ_recent_n or method_name in self.manager.layer_differ_recent_n:
            if single is not None:
                key_states,value_states,new_attn_weights = key_states_all,value_states_all,new_attn_weights_all
                bsz,num_heads,q_len,head_dim = key_states.size()
                i = single
                attn_weights = self.attn_weights[i]
                logger.debug(f"new_attn_weights.shape: {new_attn_weights.shape}")
                now_attn_weights = torch.nn.functional.pad(attn_weights, (0, 1, 0, 1), mode='constant', value=0).to(torch.float32)
                now_attn_weights[:,:,-1:,:] = new_attn_weights[...,:now_attn_weights.shape[-1]] 
                attn_weights_sum = now_attn_weights[...,-self.window_size[i]:,:-self.window_size[i]].sum(-2)
                if attn_weights_sum.shape[1] == 0:
                    self.attn_weights[i] = self.attn_weights[i] 
                    past_key_value.key_cache[layer_idx][i] = past_key_value.key_cache[layer_idx][i][:,:,:-1,:]
                    past_key_value.value_cache[layer_idx][i] = past_key_value.value_cache[layer_idx][i][:,:,:-1,:]
                cache_size = self.cache_size[i]
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')
                indices  = attn_cache.topk(cache_size-self.window_size[i], dim=-1).indices
                bsz,num_heads,q_len,_= key_states.shape
                recent_indices = self.recent_indices_generate[i]+q_len
                indices = torch.cat([indices,recent_indices],dim=-1)
                logger.debug(f"indices.shape: {indices.shape}")
                indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size[i], -1)
                logger.debug(f"self.window_size[i]: {self.window_size[i]} ")
                attn_weights_compress = now_attn_weights[...,-self.window_size[i]:,:].gather(dim = -1, index = indices_attn)
                revise_key_states = key_states.gather(dim = 2, index = indices_expanded)
                revise_value_states = value_states.gather(dim = 2, index = indices_expanded)
                self.attn_weights[i] = attn_weights_compress
                past_key_value.key_cache[layer_idx][i] = revise_key_states
                past_key_value.value_cache[layer_idx][i] = revise_value_states
                
                if revise_key_states.shape != (bsz, num_heads, cache_size, head_dim):
                    print("revise_key_states.shape",revise_key_states.shape)
                    print("max_capacity_prompt",cache_size)
                    raise ValueError("revise_key_states.shape != (bsz, num_heads, max_capacity_prompt, head_dim)")
            else:
                for i,(key_states,value_states,new_attn_weights) in enumerate(zip(key_states_all,value_states_all,new_attn_weights_all)): 
                    bsz,num_heads,q_len,head_dim = key_states.size()
                    attn_weights = self.attn_weights[i]
                    logger.debug(f"new_attn_weights.shape: {new_attn_weights.shape}")
                    now_attn_weights = torch.nn.functional.pad(attn_weights, (0, 1, 0, 1), mode='constant', value=0).to(torch.float32)
                    now_attn_weights[:,:,-1:,:] = new_attn_weights[...,:now_attn_weights.shape[-1]] 
                    attn_weights_sum = now_attn_weights[...,-self.window_size[i]:,:-self.window_size[i]].sum(-2)
                    if attn_weights_sum.shape[1] == 0:
                        self.attn_weights[i] = self.attn_weights[i] 
                        past_key_value.key_cache[layer_idx][i] = past_key_value.key_cache[layer_idx][i][:,:,:-1,:]
                        past_key_value.value_cache[layer_idx][i] = past_key_value.value_cache[layer_idx][i][:,:,:-1,:]
                        continue
                    cache_size = self.cache_size[i]
                    if self.pooling == 'avgpool':
                        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    elif self.pooling == 'maxpool':
                        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    else:
                        raise ValueError('Pooling method not supported')
                    indices  = attn_cache.topk(cache_size-self.window_size[i], dim=-1).indices
                    bsz,num_heads,q_len,_= key_states.shape
                    recent_indices = self.recent_indices_generate[i]+q_len
                    indices = torch.cat([indices,recent_indices],dim=-1)
                    logger.debug(f"indices.shape: {indices.shape}")
                    indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size[i], -1)
                    logger.debug(f"self.window_size[i]: {self.window_size[i]} ")
                    attn_weights_compress = now_attn_weights[...,-self.window_size[i]:,:].gather(dim = -1, index = indices_attn)
                    revise_key_states = key_states.gather(dim = 2, index = indices_expanded)
                    revise_value_states = value_states.gather(dim = 2, index = indices_expanded)
                    self.attn_weights[i] = attn_weights_compress
                    past_key_value.key_cache[layer_idx][i] = revise_key_states
                    past_key_value.value_cache[layer_idx][i] = revise_value_states
                    
                    if revise_key_states.shape != (bsz, num_heads, cache_size, head_dim):
                        print("revise_key_states.shape",revise_key_states.shape)
                        print("max_capacity_prompt",cache_size)
                        raise ValueError("revise_key_states.shape != (bsz, num_heads, max_capacity_prompt, head_dim)")
        elif method_name in self.manager.head_set:
            if single is not None:
                key_states,value_states,new_attn_weights = key_states_all,value_states_all,new_attn_weights_all
                i = single
                bsz,num_heads,q_len,head_dim = key_states.size()
                attn_weights = self.attn_weights[i]
                now_attn_weights = torch.nn.functional.pad(attn_weights, (0, 1, 0, 1), mode='constant', value=0).to(torch.float32)
                now_attn_weights[:,:,-1:,:] = new_attn_weights[...,:now_attn_weights.shape[-1]] 
                attn_weights_sum = now_attn_weights[...,-self.window_size:,:-self.window_size].sum(-2)
                if attn_weights_sum.shape[1] == 0:
                    self.attn_weights[i] = self.attn_weights[i] 
                    past_key_value.key_cache[layer_idx][i] = past_key_value.key_cache[layer_idx][i][:,:,:-1,:]
                    past_key_value.value_cache[layer_idx][i] = past_key_value.value_cache[layer_idx][i][:,:,:-1,:]
                cache_size = self.cache_size[i]
                # print(cache_size)
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')
                indices  = attn_cache.topk(cache_size-self.window_size, dim=-1).indices
                bsz,num_heads,q_len,_= key_states.shape
                recent_indices = self.recent_indices_generate[i]+q_len
                indices = torch.cat([indices,recent_indices],dim=-1)
                indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                attn_weights_compress = now_attn_weights[...,-self.window_size:,:].gather(dim = -1, index = indices_attn)
                revise_key_states = key_states.gather(dim = 2, index = indices_expanded)
                revise_value_states = value_states.gather(dim = 2, index = indices_expanded)
                self.attn_weights[i] = attn_weights_compress
                past_key_value.key_cache[layer_idx][i] = revise_key_states
                past_key_value.value_cache[layer_idx][i] = revise_value_states
                
                if revise_key_states.shape != (bsz, num_heads, cache_size, head_dim):
                        print("revise_key_states.shape",revise_key_states.shape)
                        print("max_capacity_prompt",cache_size)
                        raise ValueError("revise_key_states.shape != (bsz, num_heads, max_capacity_prompt, head_dim)")
            else:
                for i,(key_states,value_states,new_attn_weights) in enumerate(zip(key_states_all,value_states_all,new_attn_weights_all)): 
                    bsz,num_heads,q_len,head_dim = key_states.size()
                    attn_weights = self.attn_weights[i]
                    now_attn_weights = torch.nn.functional.pad(attn_weights, (0, 1, 0, 1), mode='constant', value=0).to(torch.float32)
                    now_attn_weights[:,:,-1:,:] = new_attn_weights[...,:now_attn_weights.shape[-1]] 
                    attn_weights_sum = now_attn_weights[...,-self.window_size:,:-self.window_size].sum(-2)
                    if attn_weights_sum.shape[1] == 0:
                        self.attn_weights[i] = self.attn_weights[i] 
                        past_key_value.key_cache[layer_idx][i] = past_key_value.key_cache[layer_idx][i][:,:,:-1,:]
                        past_key_value.value_cache[layer_idx][i] = past_key_value.value_cache[layer_idx][i][:,:,:-1,:]
                        continue
                    cache_size = self.cache_size[i]
                    if self.pooling == 'avgpool':
                        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    elif self.pooling == 'maxpool':
                        attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                    else:
                        raise ValueError('Pooling method not supported')
                    indices  = attn_cache.topk(cache_size-self.window_size, dim=-1).indices
                    bsz,num_heads,q_len,_= key_states.shape
                    recent_indices = self.recent_indices_generate[i]+q_len
                    indices = torch.cat([indices,recent_indices],dim=-1)
                    indices_expanded  = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    indices_attn = indices.unsqueeze(-2).expand(-1, -1, self.window_size, -1)
                    attn_weights_compress = now_attn_weights[...,-self.window_size:,:].gather(dim = -1, index = indices_attn)
                    revise_key_states = key_states.gather(dim = 2, index = indices_expanded)
                    revise_value_states = value_states.gather(dim = 2, index = indices_expanded)
                    self.attn_weights[i] = attn_weights_compress
                    past_key_value.key_cache[layer_idx][i] = revise_key_states
                    past_key_value.value_cache[layer_idx][i] = revise_value_states
                    
                    if revise_key_states.shape != (bsz, num_heads, cache_size, head_dim):
                            print("revise_key_states.shape",revise_key_states.shape)
                            print("max_capacity_prompt",cache_size)
                            raise ValueError("revise_key_states.shape != (bsz, num_heads, max_capacity_prompt, head_dim)")
                    
        return None
    
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

    def update_all_past_key_values(
        self,
        generate_label,
        past_key_value,
        new_key_states, 
        new_value_states,
        layer_idx,
        new_attn_weights,
    ):  
        manager = self.manager
        if generate_label == 0:
            def QS62h(entropy,manager):
                diffs = entropy[1:] - entropy[:-1]
                mask = diffs < 0
                descent_indices = torch.nonzero(mask).squeeze(0) + 1
                q_len = manager.key_list[0].shape[2]
                max_capacity_prompt = []
                j = 0
                descent_indices = descent_indices-1
                for i in range(len(descent_indices)):
                    min_num = self.max_capacity_prompt // 2
                    raw_max_num = self.max_capacity_prompt * 2 - min_num
                    if raw_max_num >= q_len:
                        raw_max_num = q_len
                        min_num = self.max_capacity_prompt * 2 - raw_max_num
                    steps = (raw_max_num - min_num) // (self.num_hidden_layers-1) #计算当前的步数
                    #计算当前层的最大cache长度
                    # descent_indices: 1 3 5 6 7 
                    if j <= descent_indices[i] and i != 0 :
                        max_num = raw_max_num - steps*(descent_indices[i-1]+1)
                        min_num = raw_max_num - steps*descent_indices[i]
                        stage_len = descent_indices[i] - descent_indices[i-1]
                        now_max = int((min_num+max_num)*stage_len //2 //stage_len)
                    else:
                        max_num = raw_max_num - steps*0
                        min_num = raw_max_num - steps*descent_indices[i]
                        stage_len = descent_indices[i] - 0 + 1
                        now_max = int((min_num+max_num)*stage_len //2 //stage_len)
                    while j <= descent_indices[i]:
                        # print("now_max",now_max)
                        max_capacity_prompt.append(now_max)
                        j += 1
                while j <=31: # 处理最后一组
                    max_num = raw_max_num - steps*(descent_indices[-1]+1)
                    min_num = raw_max_num - steps*31
                    stage_len = 31 - descent_indices[-1]
                    now_max = int((min_num+max_num)*stage_len //2 //stage_len)
                    max_capacity_prompt.append(now_max)
                    j=j+1
                max_capacity_prompt = torch.tensor(max_capacity_prompt).to(entropy.device)
                if len(max_capacity_prompt) != 32:
                    raise ValueError("The length of max_capacity_prompt_all should be 32.")
                
                return max_capacity_prompt
            manager.cache_size = []
            manager.recent_size = []
            manager.sink_size =[]
            manager.hh_size = []
            entropys = torch.from_numpy(np.array(manager.entropy)).to(new_key_states.device)
            max_capacity_prompt = [self.max_capacity_prompt]*new_key_states.size(1)
            presuppose = [128,192,192]
            if manager.method_name == "QS62h":
                max_capacity_prompt = QS62h(entropys,manager)
            
            len_key,len_value,len_attn= len(manager.key_list),len(manager.value_list),len(manager.attn_weights_sum_list)
            if len_key != len_value or len_key != len_attn:
                print("len_key",len_key)
                print("len_value",len_value)
                print("len_attn",len_attn)
                raise ValueError("The length of key_list, value_list and attn_weigths_sum_list should be the same.")
            def spearman_correlation(v1, v2):
                return stats.spearmanr(v1, v2)[0]
            def pearson_correlation(v1, v2):
                return np.corrcoef(v1, v2)[0, 1]
            def cosine_similarity(v1, v2):
                return 1 - cosine(v1, v2)
            
            if manager.method_name == "pearson_correlation_survey": # 累积注意力分数
                max_pearson_correlations = 0
                stride = 100
                max_presuppose = []
                for m in range(508//stride):
                    k = 512-m*stride-4
                    presuppose = [0,4+m*stride,k]
                    revise_key_statess = []
                    for i,(key_states,value_states,attn_weights_sum_item) in enumerate(zip(manager.key_list,manager.value_list,manager.attn_weights_sum_list)):
                        bsz,num_heads,q_len,head_dim = key_states.shape
                        max_capacity_prompt_i = max_capacity_prompt[i]
                        
                        sink_size = self.sink_size = max_capacity_prompt_i * presuppose[0] // self.max_capacity_prompt
                        hh_size = self.hh_size = max_capacity_prompt_i * presuppose[1] // self.max_capacity_prompt
                        recent_size = self.recent_size = max_capacity_prompt_i * presuppose[2] // self.max_capacity_prompt
                        
                        if q_len <= max_capacity_prompt_i:
                            sink_size = self.sink_size = q_len * presuppose[0] // self.max_capacity_prompt
                            hh_size = self.hh_size =  q_len * presuppose[1] // self.max_capacity_prompt
                            recent_size = self.recent_size = q_len * presuppose[2] // self.max_capacity_prompt
                        recent_size,hh_size,sink_size = int(recent_size),int(hh_size),int(sink_size)
                        
                        cache_size = recent_size + hh_size + sink_size
                        
                        attn_cache = attn_weights_sum_item
                        begin_recent = q_len - recent_size
                        begin_hh = sink_size
                        select_hh_scores = attn_cache[:, begin_hh:begin_recent]
                        _, keep_topk = torch.topk(select_hh_scores, hh_size, dim=-1)
                        select_sink_scores = attn_cache[:, :begin_hh]
                        _, keep_sink = torch.topk(select_sink_scores, sink_size, dim=-1)
                        select_recent_scores = attn_cache[:,begin_recent:]
                        _, keep_recent = torch.topk(select_recent_scores, recent_size, dim=-1)
                        keep_topk = keep_topk + begin_hh
                        keep_topk = keep_topk.sort().values 
                        keep_recent = keep_recent + begin_recent
                        keep_recent = keep_recent.sort().values 
                        keep_sink = keep_sink.sort().values 
                        keep_idx = torch.cat([keep_sink, keep_topk, keep_recent], dim=-1)
                        
                        mask = torch.zeros((attn_weights_sum_item.shape[0],attn_weights_sum_item.shape[1]), dtype=torch.bool).to(key_states.device)
                        mask = mask.scatter(-1, keep_idx, 1)
                        
                        revise_key_states = key_states.squeeze(0)[mask].view(bsz, num_heads, -1, head_dim)
                        revise_key_statess.append(revise_key_states)
                        if revise_key_states.shape != (bsz, num_heads, cache_size, head_dim):
                            print("revise_key_states.shape",revise_key_states.shape)
                            print("bsz, num_heads, cache_size, head_dim",(bsz, num_heads, cache_size, head_dim))
                            raise ValueError('revise_key_states shape error')
                    revise_key_statess = torch.stack(revise_key_statess,dim=2)
                    entropys_1 = get_entropy_5_dimensions_single_batch(revise_key_statess)
                    logger.info(F"entropys_1.shape:{entropys_1.shape}")
                    pearson_correlations=pearson_correlation(entropys_1,np.array(manager.entropy))
                    print(f"\npresuppose:{presuppose} pearson_correlation is {pearson_correlations}")
                    print("entropys_1",entropys_1)
                    print("manager.entropy",np.array(manager.entropy))
                    if pearson_correlations > max_pearson_correlations:
                        max_pearson_correlations = pearson_correlations
                        max_presuppose = presuppose
                
                print(f"now presuppose:{max_presuppose} pearson_correlation is {max_pearson_correlations}")
                sys.exit()
            elif manager.method_name == "pearson_correlation_survey_1": #srecent n token
                max_pearson_correlations = 0
                stride = 100
                max_presuppose = []
                for m in range(512//stride):
                    k = 512-m*stride-8
                    if k < 0:
                        continue
                    presuppose = [0,k,m*stride+8]
                    revise_key_statess = []
                    for i,(key_states,value_states,attn_weights_sum_item) in enumerate(zip(manager.key_list,manager.value_list,manager.attn_weights_sum_list)):
                        bsz,num_heads,q_len,head_dim = key_states.shape
                        max_capacity_prompt_i = max_capacity_prompt[i]
                        
                        sink_size = self.sink_size = max_capacity_prompt_i * presuppose[0] // self.max_capacity_prompt
                        hh_size = self.hh_size = max_capacity_prompt_i * presuppose[1] // self.max_capacity_prompt
                        recent_size = self.recent_size = max_capacity_prompt_i * presuppose[2] // self.max_capacity_prompt
                        
                        if m*stride+1 < self.window_size:
                            recent_size = self.recent_size = self.window_size
                            hh_size = self.hh_size = max_capacity_prompt_i - recent_size
                        if q_len <= max_capacity_prompt_i:
                            sink_size = self.sink_size = q_len * presuppose[0] // self.max_capacity_prompt
                            hh_size = self.hh_size =  q_len * presuppose[1] // self.max_capacity_prompt
                            recent_size = self.recent_size = q_len * presuppose[2] // self.max_capacity_prompt
                        self.window_size = recent_size
                        recent_size,hh_size,sink_size = int(recent_size),int(hh_size),int(sink_size)
                        
                        cache_size = recent_size + hh_size + sink_size
                        
                        attn_weights_sum = attn_weights_sum_item[:, :, -self.window_size:, :-self.window_size].sum(0).sum(1)
                        # bsz nhead seq_len-self.window_size
                        if self.pooling == 'avgpool':
                            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        elif self.pooling == 'maxpool':
                            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                        else:
                            raise ValueError('Pooling method not supported')
                        begin_recent = q_len - recent_size
                        begin_hh = sink_size
                        select_hh_scores = attn_cache[:, begin_hh:begin_recent]
                        _, keep_topk = torch.topk(select_hh_scores, hh_size, dim=-1)
                        select_sink_scores = attn_cache[:, :begin_hh]
                        _, keep_sink = torch.topk(select_sink_scores, sink_size, dim=-1)
                        select_recent_scores = attn_cache[:,begin_recent:]
                        _, keep_recent = torch.topk(select_recent_scores, recent_size-self.window_size, dim=-1)
                        keep_topk = keep_topk + begin_hh
                        keep_topk = keep_topk.sort().values 
                        keep_recent = keep_recent + begin_recent
                        keep_recent = keep_recent.sort().values 
                        keep_sink = keep_sink.sort().values 
                        keep_rrecent = torch.arange(q_len-self.window_size, q_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
                        keep_idx = torch.cat([keep_sink, keep_topk, keep_recent, keep_rrecent], dim=-1)
                        # print("keep_rrecent.shape",keep_rrecent.shape)
                        # print("keep_sink.shape",keep_sink.shape)
                        # print("keep_topk.shape",keep_topk.shape)
                        # print("keep_recent.shape",keep_recent.shape)
                        mask = torch.zeros((attn_weights_sum_item.shape[1],attn_weights_sum_item.shape[-1]), dtype=torch.bool).to(key_states.device)
                        mask = mask.scatter(-1, keep_idx, 1)
                        # print("keep_idx.shape",keep_idx.shape)
                        # print("mask.shape",mask.shape)
                        # print("mask[0,:]",mask[0,:])
                        revise_key_states = key_states.squeeze(0)[mask].view(bsz, num_heads, -1, head_dim)
                        revise_key_statess.append(revise_key_states)
                        if revise_key_states.shape != (bsz, num_heads, cache_size, head_dim):
                            print("revise_key_states.shape",revise_key_states.shape)
                            print("bsz, num_heads, cache_size, head_dim",(bsz, num_heads, cache_size, head_dim))
                            raise ValueError('revise_key_states shape error')
                    revise_key_statess = torch.stack(revise_key_statess,dim=2)
                    entropys_1 = get_entropy_head_5_dimensions_single_batch(revise_key_statess)
                    entropys_1=np.exp(entropys_1).sum(axis=0)
                    pearson_correlations=pearson_correlation(entropys_1,np.array(manager.entropy))
                    print(f"\npresuppose:{presuppose} pearson_correlation is {pearson_correlations}")
                    print("entropys_1",entropys_1)
                    print("manager.entropy",np.array(manager.entropy))
                    if pearson_correlations > max_pearson_correlations:
                        max_pearson_correlations = pearson_correlations
                        max_presuppose = presuppose
                # manager.presuppose_sum[0]+= max_presuppose[0]
                # manager.presuppose_sum[1]+= max_presuppose[1]
                # manager.presuppose_sum[2]+= max_presuppose[2]
                # if manager.pearson_correlations < max_pearson_correlations:
                #     manager.pearson_correlations = max_pearson_correlations
                #     manager.presuppose = max_presuppose
                print(f"now presuppose:{max_presuppose} pearson_correlation is {max_pearson_correlations}")
                sys.exit()
            
        elif generate_label == 1:
            past_key_value.key_cache[layer_idx]  = torch.cat([past_key_value[layer_idx][0], new_key_states], dim=-2)
            past_key_value.value_cache[layer_idx]  = torch.cat([past_key_value[layer_idx][1], new_value_states], dim=-2)
            return past_key_value.key_cache[layer_idx] ,past_key_value.value_cache[layer_idx]
        elif generate_label == 2: #更新
            sink_size,recent_size,hh_size,cache_size = manager.sink_size[layer_idx],manager.recent_size[layer_idx],manager.hh_size[layer_idx],manager.cache_size[layer_idx]
            attn_weights_sum = manager.attn_weights_sum_list[layer_idx]
            bsz,num_heads,q_len,head_dim = new_key_states.shape
            new_attn_weights = new_attn_weights.sum(0).sum(1)  
            new_attn_weights[:,:-1] += attn_weights_sum
            attn_weights_sum = new_attn_weights
                
            select_hh_scores = attn_weights_sum[:, sink_size: q_len - recent_size]
            _, keep_topk = torch.topk(select_hh_scores, hh_size, dim=-1)
            keep_topk += sink_size
            keep_topk = keep_topk.sort().values # n_head,hh_size
            
            keep_sink = torch.arange(0, sink_size, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
            keep_recent = torch.arange(q_len - recent_size, q_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
            keep_idx = torch.cat([keep_sink,keep_topk, keep_recent], dim=-1)
                
            mask = torch.zeros(attn_weights_sum.shape, dtype=torch.bool).to(attn_weights_sum.device)
            mask = mask.scatter_(-1, keep_idx, 1)
                
            past_key_value.key_cache[layer_idx] = new_key_states.squeeze(0)[mask].view(bsz, num_heads, -1, head_dim)
            past_key_value.value_cache[layer_idx] = new_value_states.squeeze(0)[mask].view(bsz, num_heads, -1, head_dim)
                
                
            manager.attn_weights_sum_list[layer_idx]= attn_weights_sum[mask].view(num_heads, cache_size)
        return None,None
    

def init_uncomp(self, num_hidden_layers,manager):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 512
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    
    self.kv_cluster = UncompCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        manager = manager,
        )
