import os
import json
import random
import argparse
import sys
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import re
from uncomp.utils.logger import Logger
import logging

# 创建logger实例

accelerator = Accelerator()
logger = Logger()

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "trec", "triviaqa","passage_count", "passage_retrieval_en",  \
            "qmsum","samsum","lcc", "repobench-p","gov_report","multi_news"] 
datasets = ["multifieldqa_en"]
dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "qasper_new": 128,
    "multifieldqa_en": 64,
    "multifieldqa_en_e": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "hotpotqa_new": 32,
    "2wikimqa": 32,
    "2wikimqa_new": 32,
    "musique": 32,
    # "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    # "vcsum": 512,
    "trec": 64,
    "trec_new": 64,
    "triviaqa": 32,
    "triviaqa_new": 32,
    # "triviaqa_e": 32,
    "samsum": 128,
    "samsum_new": 128,
    # "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_en_new": 32,
    # "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
    "repobench-p_new": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper_new": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_en_e": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa_new": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa_new": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "trec_new": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    # "triviaqa_e": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "triviaqa_new": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "samsum_new": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_en_new": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    "repobench-p_new": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 3950,
    "mistral": 12000,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt

def main(args,manager):
    logger.info("Loading data...")
    test_data = []
    prompts_all = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    input_max_len = 0
    
    model_path = args.model_path.lower()
    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
 
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            length = example["length"]
            if length > input_max_len: input_max_len = length
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
            example["prompt"] = prompt
            test_data.append(example)
        
    logger.info(f"Max Length is {input_max_len}")
    
    for example in test_data:
        prompts_all.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    logger.info("Finish loading model and tokenizer")
    
    combined_data = [
        {
            "prompt": p, "input": i, "context": c, "answers": a, 
            "length": l, "dataset": d, "language": lang, 
            "all_classes": ac, "_id": id
        }
        for p, i, c, a, l, d, lang, ac, id in zip(
            prompts_all, inputs, contexts, answerss, lengths, 
            datasets, languages, all_classess, _ids
        )
    ]
    
    if manager.method_name in manager.rope_correlation:
        combined_data = combined_data[:1]
        output_max_len = 1
        manager.rope_correlation_dict["dataname"] = args.dataset
    
    manager.max_used = 0
    manager.min_used = torch.iinfo(torch.long).max
    start=time.time()
    with accelerator.split_between_processes(combined_data) as split_data:
        results=dict(outputs=[], num_tokens=0, first_token_time=0)
        manager.sum1 = 0
        split_data = list(split_data)
        
        for i in tqdm(range(0, len(split_data), args.eval_batch_size)):
            batch_data = split_data[i:i+args.eval_batch_size]
        
            batch_prompts = [item["prompt"] for item in batch_data]
            batch_inputs = [item["input"] for item in batch_data]
            batch_contexts = [item["context"] for item in batch_data]
            batch_answerss = [item["answers"] for item in batch_data]
            batch_lengths = [item["length"] for item in batch_data]
            batch_datasets = [item["dataset"] for item in batch_data]
            batch_languages = [item["language"] for item in batch_data]
            batch_all_classess = [item["all_classes"] for item in batch_data]
            batch__ids = [item["_id"] for item in batch_data]
            tokenized_prompts = tokenizer(batch_prompts, 
                                          padding="longest", 
                                          return_tensors="pt", 
                                          add_special_tokens=True,
                                          ).to('cuda')
            
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask
            actual_lengths = attention_mask.sum(dim=1)
            max_len = actual_lengths.max().item()
            padding_len = max_len - actual_lengths
            if args.eval_batch_size == 1:
                if len(batch_input_ids[0]) > model_max_len:
                    half = int(model_max_len/2)
                    prompt = [tokenizer.decode(batch_input_ids[i][padding_len[i]:padding_len[i]+half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[i][-half:], skip_special_tokens=True) for i in range(len(batch_input_ids))]
                    tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
                    batch_input_ids = tokenized_prompts.input_ids
                    attention_mask = tokenized_prompts.attention_mask
            else:            
                if len(batch_input_ids[0]) > model_max_len:
                    new_batch_input_ids = torch.zeros((len(batch_input_ids), model_max_len), dtype=torch.long).to('cuda')
                    new_attention_mask = torch.zeros((len(attention_mask), model_max_len), dtype=torch.long).to('cuda')
                    half = int(model_max_len/2)
                    for i in range(len(batch_input_ids)):
                        new_batch_input_ids[i] = torch.cat([batch_input_ids[i,padding_len[i]:padding_len[i]+half],batch_input_ids[i,-half:]],dim=0)
                        new_attention_mask[i] = torch.cat([attention_mask[i,padding_len[i]:padding_len[i]+half],attention_mask[i,-half:]],dim=0)
                    batch_input_ids = new_batch_input_ids
                    attention_mask = new_attention_mask
            if args.max_capacity_prompts != -1:
                max_capacity_prompts = args.max_capacity_prompts

            if args.method != "FullKV":
                window_sizes = 8
                if "wind" in args.method_name:
                    match = re.search(r'wind(\d+)', args.method_name)
                    window_sizes = int(match.group(1))
                kernel_sizes = 7
                pooling = "maxpool"
                layers = len(model.model.layers)
                if not isinstance(window_sizes, list):
                    window_sizes = [window_sizes] * layers
                if not isinstance(max_capacity_prompts, list):
                    max_capacity_prompts = [max_capacity_prompts] * layers
                if not isinstance(kernel_sizes, list):
                    kernel_sizes = [kernel_sizes] * layers
                for i in range(layers):
                    model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                    model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                    model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                    model.model.layers[i].self_attn.config.pooling = pooling
            context_length = batch_input_ids.shape[-1]
            
            if manager.calib_label:
                manager.sample_time = 0
                datas = manager.calib_datas
                manager.presuppose = [0,0,0]
                manager.presuppose_sum = [0,0,0]
                manager.pearson_correlations = 0
                for i in range(len(datas)):
                    model.generate(
                        **datas[i],
                        output_attentions = args.output_attentions,
                        max_new_tokens=1,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=4096+1,
                        eos_token_id=[tokenizer.eos_token_id]
                    )
                    manager.sample_time+=1
                    print("manager.sample_time",manager.sample_time)
                sys.exit(0)
            
            output = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id]
            )
            torch.cuda.empty_cache()
            func_utils_instance = manager.func_utils_instance
            get_memory_info = func_utils_instance.get_memory_info
            bytes_to_gb = func_utils_instance.bytes_to_gb
            reserved, allocated = get_memory_info()
            print(f"Memory freed:        reserved={bytes_to_gb(reserved)} GB, allocated={bytes_to_gb(allocated)} GB")
            # assert 1==0
            batch_outputs =tokenizer.batch_decode(output[:,context_length:], skip_special_tokens=True)
            print("\nbatch_outputs is ", batch_outputs)
            
            batch_generations = batch_outputs
            for j in range(len(batch_prompts)):
                example = {}
                example["prompt"] = batch_prompts[j]
                example["input"] = batch_inputs[j]
                example["context"] = batch_contexts[j]
                example["answers"] = batch_answerss[j]
                example["pred"] = batch_generations[j]
                example["length"] = batch_lengths[j]
                
                example["dataset"] = batch_datasets[j]
                example["language"] = batch_languages[j]
                example["all_classes"] = batch_all_classess[j]
                example["_id"] = batch__ids[j]
                results["outputs"].append(example)
                results["num_tokens"] += len(batch_generations[j])
            # def get_rocm_memory():
            #     result = subprocess.run(
            #         ['rocm-smi','--showmeminfo', 'vram'],
            #         stdout=subprocess.PIPE,
            #         stderr=subprocess.PIPE,
            #         text=True
            #     )
            #     lines = result.stdout.splitlines()
            #     for i, line in enumerate(lines):
            #         if "GPU[2]" in line and "VRAM Total Used Memory (B)" in line:
            #             value = int(line.split(": ")[2].strip().split()[0])
            #             value = value /(1024**2)
            #             # print(f"GPU 2 VRAM Total Memory : {value :.2f} MB")
            #             return value
            
            # used = get_rocm_memory()
            # if manager.max_used < used:
            #     manager.max_used = used
            # if manager.min_used > used:
            #     manager.min_used = used
            # print(f'Max used: {manager.max_used:.2f} MB, Min used: {manager.min_used:.2f} MB')
        results = [results]
        sums = [manager.sum1]
    
    results_gathered = gather_object(results)
    accelerator.wait_for_everyone()
    sum_gathered = gather_object(sums)
    if accelerator.is_main_process:
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])
        model_name = model_path.split("/")[-1]
        os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset), exist_ok=True)
        fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}.json"), "w")
        for result_list in results_gathered:
            for example in result_list["outputs"]:
                fout.write(json.dumps(example) + "\n")
        print("sum(sum_gathered)", sum(sum_gathered))
        print("mean compress", sum(sum_gathered)/len(combined_data)/(manager.num_hidden_layers*manager.num_attention_heads))
        print(f"all sec/token: {timediff / num_tokens}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")
 
def calib_main(args,manager):
    logger.info("Loading data...")
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
    
    model_max_lens = [2048,8192]
    test_data = manager.calib_datas
    manager.max_used = 0
    manager.min_used = torch.iinfo(torch.long).max
    start=time.time()
    with accelerator.split_between_processes(test_data) as split_data:
        results=dict(outputs=[], num_tokens=0, first_token_time=0, prompt = 0)
        manager.sum1 = 0
        split_data = list(split_data)
        logger.debug(f"len(split_data): {len(split_data)}")
        for i in tqdm(range(0, len(split_data), args.eval_batch_size)):
            batch_data = split_data[i:i+args.eval_batch_size]
            
            tokenized_prompts = tokenizer(batch_data, 
                                          padding="longest", 
                                          return_tensors="pt", 
                                          add_special_tokens=True,
                                          ).to('cuda')
            
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask
            print("attnention_mask.dtype",attention_mask.dtype)
            print("batch_input_ids.dtype",batch_input_ids.dtype)
            actual_lengths = attention_mask.sum(dim=1)
            max_len = actual_lengths.max().item()
            padding_len = max_len - actual_lengths
            
            if args.eval_batch_size == 1:
                if len(batch_input_ids[0]) > model_max_lens[0]:
                    half = int(model_max_lens[0]/2)
                    prompt = [tokenizer.decode(batch_input_ids[i][padding_len[i]:padding_len[i]+half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[i][-half:], skip_special_tokens=True) for i in range(len(batch_input_ids))]
                    tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
                    batch_input_ids = tokenized_prompts.input_ids
                    attention_mask = tokenized_prompts.attention_mask
            else:            
                if len(batch_input_ids[0]) > model_max_lens[0]:
                    new_batch_input_ids = torch.zeros((len(batch_input_ids), model_max_lens[0]), dtype=torch.long).to('cuda')
                    new_attention_mask = torch.zeros((len(attention_mask), model_max_lens[0]), dtype=torch.long).to('cuda')
                    half = int(model_max_lens[0]/2)
                    for i in range(len(batch_input_ids)):
                        new_batch_input_ids[i] = torch.cat([batch_input_ids[i,padding_len[i]:padding_len[i]+half],batch_input_ids[i,-half:]],dim=0)
                        new_attention_mask[i] = torch.cat([attention_mask[i,padding_len[i]:padding_len[i]+half],attention_mask[i,-half:]],dim=0)
                    batch_input_ids = new_batch_input_ids
                    attention_mask = new_attention_mask
            max_capacity_prompts = args.max_capacity_prompts
            if args.method != "FullKV":
                window_sizes = 8
                kernel_sizes = 7
                pooling = "maxpool"
                layers = len(model.model.layers)
                if not isinstance(window_sizes, list):
                    window_sizes = [window_sizes] * layers
                if not isinstance(max_capacity_prompts, list):
                    max_capacity_prompts = [max_capacity_prompts] * layers
                if not isinstance(kernel_sizes, list):
                    kernel_sizes = [kernel_sizes] * layers
                for i in range(layers):
                    model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                    model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                    model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                    model.model.layers[i].self_attn.config.pooling = pooling

            context_length = batch_input_ids.shape[-1]
            print("tokenized_prompts.input_ids.shape", tokenized_prompts.input_ids.shape)
            
            manager.sample_time = 0
            manager.presuppose = [0,0,0]
            manager.presuppose_sum = [0,0,0]
            manager.pearson_correlations = 0
            
            manager.bsz = batch_input_ids.shape[0]

            output = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                output_attentions = args.output_attentions,
                max_new_tokens=model_max_lens[1],
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=model_max_lens[0]+model_max_lens[1],
                eos_token_id=[tokenizer.eos_token_id]
            )
            logger.debug(f"output.shape: {output.shape}")
            
            results["num_tokens"] += output[:,context_length:].shape[-1]*batch_input_ids.shape[0]
            results["prompt"] += batch_input_ids.shape[-1]*batch_input_ids.shape[0]
            
            def get_rocm_memory():
                result = subprocess.run(
                    ['rocm-smi','--showmeminfo', 'vram'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                lines = result.stdout.splitlines()
                for i, line in enumerate(lines):
                    if "GPU[2]" in line and "VRAM Total Used Memory (B)" in line:
                        value = int(line.split(": ")[2].strip().split()[0])
                        value = value /(1024**2)
                        print(f"GPU 2 VRAM Total Memory : {value :.2f} MB")
                        return value
            
            used = get_rocm_memory()
            if manager.max_used < used:
                manager.max_used = used
            if manager.min_used > used:
                manager.min_used = used
            print(f'Max used: {manager.max_used:.2f} MB, Min used: {manager.min_used:.2f} MB')
        results = [results]
        sums = [manager.sum1]
    
    results_gathered = gather_object(results)
    accelerator.wait_for_everyone()
    sum_gathered = gather_object(sums)
    if accelerator.is_main_process:
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])
        prompts = sum([r["prompt"] for r in results_gathered ])
        model_name = model_path.split("/")[-1]
        os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset), exist_ok=True)
        fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}.json"), "w")
        for result_list in results_gathered:
            for example in result_list["outputs"]:
                fout.write(json.dumps(example) + "\n")
        print(f"\n\nall sec/token: {timediff / num_tokens*1000} ms, time {timediff}, total tokens {num_tokens}, total prompts {prompts}")
        
@dataclass
class Manager:
    last_attn = None
    head_datas = [0]*100
    head_granularity,head_set,last_process,delet_head_set,group_sampling = [],[],[],[],[]
    chai_layers_llama2: list = field(default_factory=lambda: [
            32,32,30,30,
            30,30,30,30,
            24,30,24,30,
            28,28,32,32,
            18,18,20,28,
            18,18,18,20,
            18,18,18,18,
            18,18,24,32,
        ])
    chai_layers_llama2_2: list = field(default_factory=lambda: [
            32,32,32,32,
            28,28,28,28,
            20,20,20,20,
            20,20,20,20,
            8,8,8,8,
            8,8,8,8,
            8,8,8,8,
            12,12,12,12,
        ])
    chai_layers_llama2_13B: list = field(default_factory=lambda: [
            32,32,30,30,
            32,32,30,30,
            32,32,30,30,
            30,30,30,30,
            24,30,24,30,
            28,28,32,32,
            18,18,20,28,
            18,18,18,20,
            18,18,18,18,
            18,18,24,32,
        ])
    chai_layers_llama3: list = field(default_factory=lambda: [
            32,32,30,30,
            32,32,32,32,
            24,30,24,30,
            28,28,32,32,
            26,26,26,26,
            26,26,26,26,
            26,26,26,26,
            26,26,26,30,
        ])
    pass

from datasets import load_dataset
def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=4096, seed=43):
    logger.info(f"get_calib_data {name}, nsamples={nsamples}, seqlen={seqlen}, seed={seed}")
    if name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    logger.info(f"traindata[\"text\"] {type(traindata['text'])}")
    logger.info(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset

class func_utils:

    def init_manager(self,manager):
        model_path = manager.model_path
        if "llama-3" in model_path:
            manager.chai_layers = manager.chai_layers_llama3
            manager.max_token = 8192
            manager.num_hidden_layers = 32
            manager.num_attention_heads = 32
        elif "llama-2" in model_path and "13b" in model_path:
            manager.chai_layers = manager.chai_layers_llama2_13B
            manager.num_hidden_layers = 40
            manager.num_attention_heads = 40
            manager.max_token = 4096
        else:
            manager.chai_layers = manager.chai_layers_llama2
            manager.max_token = 4096
            manager.num_attention_heads = 32
            manager.num_hidden_layers = 32
    
    def last_process(self,manager):
        manager.search=["pearson_correlation_survey_1"]
        manager.last_process.extend(["pearson_correlation_survey_1","pearson_correlation_survey"])

    def rope_survey(self,manager):
        manager.rope=[]
        manager.rope_survey = []
        manager.rope_clip_layer_single_layer = []
        manager.rope_clip_layer_multi_layer = []
        manager.rope_correlation = []
        manager.rope_position_ids_control = []
        
        def get_last_number(string):
            match = re.search(r'(\d+)(?!.*\d)', string)
            if match:
                last_number = int(match.group(1))
                print(f"The last number in the string is: {last_number}")
            else:
                print("No number found in the string.")
            return last_number
        
        def get_all_numbers(string):
            matches = re.findall(r'\d+(?:\.\d+)?', string)
            if matches:
                numbers = [float(num) for num in matches]  
                print(f"All numbers in the string are: {numbers}")
            else:
                print("No numbers found in the string.")
            return numbers
        
        def get_last_float_number(string):
            match = re.search(r'(\d+(\.\d+)?)(?!.*\d)', string)
            if match:
                last_number = float(match.group(1))  
                print(f"The last number in the string is: {last_number}")
            else:
                print("No number found in the string.")
            return last_number

           
    def not_update(self,manager):
        manager.not_update = [] 
        if "not_update" in manager.method_name:
            logger.warning("kv cache is not maintained in decoding stage")
            manager.not_update.append(manager.method_name)

    def other_methods(self,manager):
        manager.chai = ["chai"]
        if "pyramidkv_generate" in manager.method_name or "snapkv" in manager.method_name:
            manager.group_sampling.append(manager.method_name)
    
    def calib_sets(self,manager):
        manager.calib = ["head_type_search_2","head_type_search_2_nosvd",
                         "head_type_search_3",
                         "head_type_search_4","head_type_search_4_nosvd",
                         "head_type_search_5",
                         "head_type_search_8", "head_type_search_8_nosvd",
                         "head_type_search_32","output_entropy",
                         "head_type_search_2_variance"]
        if "calib" in manager.method_name:
            manager.calib.append(manager.method_name)
   
    def get_memory_info(self):
        stats = torch.cuda.memory_stats()
        reserved = stats.get("reserved_bytes.all.current", 0)
        allocated = stats.get("allocated_bytes.all.current", 0)
        return reserved, allocated

    def bytes_to_gb(self,x):
        return round(x / (1024 ** 3), 4)
    
    def uncomp_sets(self,manager):
        manager.ahead_500_similar = []
        manager.ahead_500 = []
        
        manager.delete_head_equal_code = []
        if "uncomp_delete_head" in manager.method_name:
            logger.info("uncomp_delete_head")
            manager.delete_head_equal_code.append(manager.method_name)
        
        manager.extreme_compressibility_equal_code = []
        if "uncomp_extreme_compressibility" in manager.method_name:
            logger.info("uncomp_extreme_compressibility")
            manager.extreme_compressibility_equal_code.append(manager.method_name)
        
        manager.hidden_delete_stage_and_ours = []
        manager.ahead_500_equal_code = []
        if "stage" in manager.method_name:
            logger.info("uncomp_stage")
            manager.hidden_delete_stage_and_ours.append(manager.method_name)
            if "group" not in manager.method_name and "stage_only" not in manager.method_name:
                manager.ahead_500.append(manager.method_name)
                manager.ahead_500_equal_code.append(manager.method_name)
        
        if "uncomp" in manager.method_name and "extreme_compressibility" not in manager.method_name \
        and "group" not in manager.method_name and "delete_head" not in manager.method_name \
            and "pyramidkv" not in manager.method_name:
            logger.info("uncomp")
            manager.ahead_500_equal_code.append(manager.method_name)
            manager.ahead_500.append(manager.method_name)
        
        lists_to_extend = [manager.extreme_compressibility_equal_code,
                           manager.delete_head_equal_code,
                           manager.ahead_500_equal_code,
                           ]
        for lst in lists_to_extend:
            manager.head_set.extend(lst)
            manager.head_granularity.extend(lst)
    
    def uncomp_extend(self,manager):
        manager.head_differ_recent_n = []
        if "uncomp_head_group_differ_recent_n" in manager.method_name:
            logger.info("uncomp_head_group_differ_recent_n")
            manager.head_differ_recent_n.append(manager.method_name)
            manager.head_set.extend(manager.head_differ_recent_n)
            manager.head_granularity.extend(manager.head_differ_recent_n)
            manager.ahead_500.extend(manager.head_differ_recent_n)
        
        manager.layer_differ_recent_n = []
        if "uncomp_layer_group_differ_recent_n" in manager.method_name:
            logger.info("uncomp_layer_group_differ_recent_n")
            if "set1" in manager.method_name:
                manager.layer_window = list(range(1, 33))
            elif "set2" in manager.method_name:
                manager.layer_window = list(range(32, 0, -1))
            manager.layer_differ_recent_n.append(manager.method_name)
            manager.head_set.extend(manager.layer_differ_recent_n)
            manager.head_granularity.extend(manager.layer_differ_recent_n)
            manager.ahead_500.extend(manager.layer_differ_recent_n)

    def multi_group_sets(self,manager):
        manager.ahead_500_equal_code_little_size = []
        if "uncomp_little_size" in manager.method_name and "group" not in manager.method_name:
            manager.ahead_500.append(manager.method_name)
            manager.ahead_500_equal_code_little_size.append(manager.method_name)
            if "hidden_stage_uncomp_little_size" in manager.method_name and "group" not in manager.method_name:
                manager.hidden_delete_stage_and_ours.append(manager.method_name)
        lists_to_extend = [manager.ahead_500_equal_code_little_size]
        for lst in lists_to_extend:
            manager.head_set.extend(lst)
            manager.head_granularity.extend(lst)

        manager.ahead_500_group3 = []
        if "uncomp_group3" in manager.method_name and "uncomp_group32" not in manager.method_name:
            manager.ahead_500_group3.append(manager.method_name)
        manager.ahead_500_equal_code_group3=[]
        lists_to_extend = [manager.ahead_500_group3]
        for lst in lists_to_extend:
            manager.head_set.extend(lst)
            manager.head_granularity.extend(lst)
            manager.ahead_500_equal_code_group3.extend(lst)
        
        manager.ahead_500_group4 = []
        if "uncomp_group4" in manager.method_name:
            manager.ahead_500_group4.append(manager.method_name)
        manager.ahead_500_equal_code_group4=[]
        lists_to_extend = [manager.ahead_500_group4]
        for lst in lists_to_extend:
            manager.head_set.extend(lst)
            manager.head_granularity.extend(lst)
            manager.ahead_500_equal_code_group4.extend(lst)
       
        manager.ahead_500_group5 = []
        if "uncomp_group5" in manager.method_name:
            manager.ahead_500_group5.append(manager.method_name)
        manager.ahead_500_equal_code_group5=[]
        lists_to_extend = [manager.ahead_500_group5]
        for lst in lists_to_extend:
            manager.head_set.extend(lst)
            manager.head_granularity.extend(lst)
            manager.ahead_500_equal_code_group5.extend(lst)
        
        manager.ahead_500_group8 = []
        if "uncomp_group8" in manager.method_name:
            manager.ahead_500_group8.append(manager.method_name)
        manager.ahead_500_equal_code_group8=[]
        lists_to_extend = [manager.ahead_500_group8]
        for lst in lists_to_extend:
            manager.head_set.extend(lst)
            manager.head_granularity.extend(lst)
            manager.ahead_500_equal_code_group8.extend(lst)
        
        manager.ahead_500_group32 = []
        if "uncomp_group32" in manager.method_name:
            manager.ahead_500_group32.append(manager.method_name)
            manager.layer_groups = [2,2,2,2,2,2,2,2,
                                    1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,
                                    2,2,2,2,2,2,2,2,]
        manager.ahead_500_equal_code_group32=[]
        lists_to_extend = [manager.ahead_500_group32]
        for lst in lists_to_extend:
            manager.head_set.extend(lst)
            manager.head_granularity.extend(lst)
            manager.ahead_500_equal_code_group32.extend(lst)
        
        manager.pyramidkv_uncomp = []
        if "pyramidkv" in manager.method_name and "uncomp" in manager.method_name and "stage" not in manager.method_name:
            logger.info("pyramidkv_uncomp")
            if "group4" in manager.method_name:
                manager.ahead_500_group4.append(manager.method_name)
                manager.pyramidkv_uncomp_new = manager.ahead_500_group4
            
            elif "group8" in manager.method_name:
                manager.ahead_500_group8.append(manager.method_name)
                manager.pyramidkv_uncomp_new = manager.ahead_500_group8
            elif "group32" in manager.method_name:
                manager.layer_groups = [8,8,8,8,8,8,8,8,
                                        4,4,4,4,4,4,4,4,
                                        2,2,2,2,2,2,2,2,
                                        1,1,1,1,1,1,1,1]
                manager.ahead_500_group32.append(manager.method_name)
                manager.pyramidkv_uncomp_new = manager.ahead_500_group32
            else:        
                manager.ahead_500.append(manager.method_name)
                manager.pyramidkv_uncomp_new = manager.ahead_500
            manager.pyramidkv_uncomp.append(manager.method_name)
        else:
            manager.pyramidkv_uncomp_new= manager.ahead_500
                
        lists_to_extend = [manager.pyramidkv_uncomp_new]
        for lst in lists_to_extend:
            manager.head_set.extend(lst)
            manager.head_granularity.extend(lst)
            manager.ahead_500_equal_code_group32.extend(lst)
        
        
        
        manager.ahead_500.extend(manager.extreme_compressibility_equal_code)
        manager.ahead_500_similar.extend(manager.delete_head_equal_code)
        manager.delet_head_set.extend(manager.delete_head_equal_code)
    
    def determine_head_type(self,manager,device):
        if manager.method_name in manager.ahead_500:
            data_all_layers = []
            data_all_layers_2 = []
            logger.info(f"args.model_path is {args.model_path}")
            for i in range(manager.num_hidden_layers):
                if "llama-3" in manager.model_path:
                    logger.info("llama3")
                    filename = f"./search/llama3-instruct/2_groups/svd32/head_type_search_layer" + str(i) + ".csv"
                elif "variance" in manager.method_name:
                    filename = "./search/512/llama2-chat/variance/head_type_search_layer" + str(i) + ".csv"
                    logger.info(f"filename is {filename}")
                elif "uncomp" in manager.method_name:
                    if "nosvd" in manager.method_name:
                        filename = f"./search/llama2-chat/2_groups_nosvd/svd128/head_type_search_layer" + str(i) + ".csv"
                    else:
                        filename = "./search/512/llama2-chat/query/head_type_search_layer" + str(i) + ".csv"
                data_layers = []
                if os.path.isfile(filename):
                    import csv
                    with open(filename, 'r', newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            data_layers.append([int(value) for value in row])
                else:
                    logger.error("load error")
                    raise ValueError
                data_layers = np.array(data_layers)
                data_layers = data_layers.sum(axis=0)
                num_heads = manager.num_attention_heads // 2
                top_half_indices = np.argpartition(data_layers, -num_heads)[-num_heads:]
                down_half_indices = np.argpartition(data_layers, -num_heads)[:num_heads]
                indices = torch.cat([torch.tensor(down_half_indices).sort()[0],torch.tensor(top_half_indices).sort()[0]])
                if "reverse" in manager.method_name:
                    indices = torch.cat([torch.tensor(top_half_indices).sort()[0],torch.tensor(down_half_indices).sort()[0]])
                data_all_layers.append([top_half_indices,down_half_indices])
                data_all_layers_2.append(indices)
            manager.head_datas = torch.from_numpy(np.array(data_all_layers_2)).to(device)   
        elif manager.method_name in manager.ahead_500_group3:
            data_all_layers_2 = []
            for i in range(manager.num_hidden_layers):
                numbers = 32
                filename = f"./search/llama2-chat/3_groups/svd{numbers}/head_type_search_layer" + str(i) + ".csv"
                data_layers = []
                if os.path.isfile(filename):
                    import csv
                    with open(filename, 'r', newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            data_layers.append([int(value) for value in row])
                else:
                    logger.error("load error")
                    raise ValueError
                data_layers = np.array(data_layers)
                counts = np.array([np.bincount(data_layers[:, i], minlength=3) for i in range(32)]).T
                zero_indices = np.argsort(counts[0])[-11:][::-1]
                
                remaining_indices = np.setdiff1d(np.arange(32), zero_indices)
                one_indices = remaining_indices[np.argsort(counts[1, remaining_indices])[-10:][::-1]]
                
                final_indices = np.setdiff1d(np.arange(32), np.concatenate((zero_indices, one_indices)))
                indices = torch.cat([torch.from_numpy(zero_indices.copy()).sort()[0],torch.from_numpy(one_indices.copy()).sort()[0],torch.from_numpy(final_indices.copy()).sort()[0]])
                data_all_layers_2.append(indices)
            manager.head_datas = torch.from_numpy(np.array(data_all_layers_2)).to(device)
        elif manager.method_name in manager.ahead_500_group4:
            data_all_layers_2 = []
            for i in range(manager.num_hidden_layers):
                logger.info(f"manager.model_path is {manager.model_path}")
                if "llama-3" in manager.model_path:
                    logger.info("llama3")
                    filename = f"./search/llama3-instruct/4_groups/svd32/head_type_search_layer" + str(i) + ".csv"
                elif "nosvd" in manager.method_name:
                    filename = f"./search/llama2-chat/4_groups_nosvd/svd128/head_type_search_layer" + str(i) + ".csv"
                else:
                    filename = f"./search/llama2-chat/4_groups/svd{numbers}/head_type_search_layer" + str(i) + ".csv"
                data_layers = []
                if os.path.isfile(filename):
                    import csv
                    with open(filename, 'r', newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            data_layers.append([int(value) for value in row])
                else:
                    logger.error("load error")
                    raise ValueError
                data_layers = np.array(data_layers)
                
                def count_occurrences(arr):
                    counts = [[0, 0, 0, 0] for _ in range(32)]
                    for row in arr:
                        for i, value in enumerate(row):
                            counts[i][value] += 1
                    return counts

                def get_top_indices(counts, n, value):
                    indexed_counts = list(enumerate(counts))
                    sorted_counts = sorted(indexed_counts, key=lambda x: x[1][value], reverse=True)
                    return [index for index, _ in sorted_counts[:n]]
                
                counts = count_occurrences(data_layers)
                remaining_indices = list(range(32))
                results = []
                for value in range(4):
                    top_indices = get_top_indices([counts[i] for i in remaining_indices], 8, value)
                    selected_indices = [remaining_indices[i] for i in top_indices]
                    results.append(selected_indices)
                    
                    if value < 3:
                        remaining_indices = [i for i in remaining_indices if i not in selected_indices]
                indices = torch.cat([
                    torch.tensor(results[0]).sort()[0],
                    torch.tensor(results[1]).sort()[0],
                    torch.tensor(results[2]).sort()[0],
                    torch.tensor(results[3]).sort()[0]
                ])
                if "reverse" in manager.method_name:
                    indices = torch.cat([
                        torch.tensor(results[3]).sort()[0],
                        torch.tensor(results[2]).sort()[0],
                        torch.tensor(results[1]).sort()[0],
                        torch.tensor(results[0]).sort()[0]
                    ])
                    
                data_all_layers_2.append(indices)
            manager.head_datas = torch.from_numpy(np.array(data_all_layers_2)).to(device)
        elif manager.method_name in manager.ahead_500_group5:
            data_all_layers_2 = []
            for i in range(manager.num_hidden_layers):
                numbers = 32
                filename = f"./search/llama2-chat/5_groups/svd{numbers}/head_type_search_layer" + str(i) + ".csv"
                data_layers = []
                if os.path.isfile(filename):
                    import csv
                    with open(filename, 'r', newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            data_layers.append([int(value) for value in row])
                else:
                    logger.error("load error")
                    raise ValueError
                def count_occurrences(arr):
                    counts = [[0, 0, 0, 0, 0] for _ in range(32)]
                    for row in arr:
                        for i, value in enumerate(row):
                            counts[i][value] += 1
                    return counts

                def get_top_indices(counts, n, value):
                    indexed_counts = list(enumerate(counts))
                    sorted_counts = sorted(indexed_counts, key=lambda x: x[1][value], reverse=True)
                    return [index for index, _ in sorted_counts[:n]]
                
                counts = count_occurrences(data_layers)
                remaining_indices = list(range(32))
                results = []
                selection_counts = [7, 6, 6, 6, 7]
                for value, n in enumerate(selection_counts):
                    top_indices = get_top_indices([counts[i] for i in remaining_indices], n, value)
                    selected_indices = [remaining_indices[i] for i in top_indices]
                    results.append(selected_indices)
                    
                    if value < 4:
                        remaining_indices = [i for i in remaining_indices if i not in selected_indices]
                indices = torch.cat([
                    torch.tensor(results[0]).sort()[0],
                    torch.tensor(results[1]).sort()[0],
                    torch.tensor(results[2]).sort()[0],
                    torch.tensor(results[3]).sort()[0],
                    torch.tensor(results[4]).sort()[0],
                ])
                data_all_layers_2.append(indices)
            manager.head_datas = torch.from_numpy(np.array(data_all_layers_2)).to(device)
        elif manager.method_name in manager.ahead_500_group8:
            data_all_layers_2 = []
            # assert 1==0
            for i in range(manager.num_hidden_layers):
                numbers = 32
                if "llama-3" in manager.model_path:
                    logger.info("llama3")
                    filename = f"./search/llama3-instruct/8_groups/svd32/head_type_search_layer" + str(i) + ".csv"
                elif "nosvd" in manager.method_name:
                    filename = f"./search/llama2-chat/8_groups_nosvd/svd128/head_type_search_layer" + str(i) + ".csv"
                elif "13b" in manager.model_path:
                    logger.info("llama2-chat-13B")
                    filename = f"./search/llama2-chat-13B/8_groups/svd{numbers}/head_type_search_layer" + str(i) + ".csv"
                elif "mistral" in manager.model_path:
                    logger.info("Mistral")
                    filename = f"./search/mistral/8_groups/svd{numbers}/head_type_search_layer" + str(i) + ".csv"
                else:
                    # assert 1==0
                    filename = f"./search/llama2-chat/8_groups/svd{numbers}/head_type_search_layer" + str(i) + ".csv"
                data_layers = []
                if os.path.isfile(filename):
                    import csv
                    with open(filename, 'r', newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            data_layers.append([int(value) for value in row])
                else:
                    logger.error("load error")
                    raise ValueError
                
                if "13b" not in manager.model_path:
                    def count_occurrences(arr):
                        counts = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(32)]
                        for row in arr:
                            for i, value in enumerate(row):
                                counts[i][value] += 1
                        return counts

                    def get_top_indices(counts, n, value):
                        indexed_counts = list(enumerate(counts))
                        sorted_counts = sorted(indexed_counts, key=lambda x: x[1][value], reverse=True)
                        return [index for index, _ in sorted_counts[:n]]
                    
                    counts = count_occurrences(data_layers)
                    remaining_indices = list(range(32))
                    results = []
                    selection_counts = [4, 4, 4, 4, 4, 4, 4, 4]   
                    
                    for value in range(8): 
                        n = selection_counts[value]
                        top_indices = get_top_indices([counts[i] for i in remaining_indices], n, value)
                        selected_indices = [remaining_indices[i] for i in top_indices]
                        results.append(selected_indices)
                        
                        if value < 7:  
                            remaining_indices = [i for i in remaining_indices if i not in selected_indices]
                    indices = torch.cat([torch.tensor(result).sort()[0] for result in results])
                    indices = torch.cat([torch.tensor(result).sort()[0] for result in results])
                    
                    data_all_layers_2.append(indices)
                else:
                    def count_occurrences(arr):
                        counts = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(40)]
                        for row in arr:
                            for i, value in enumerate(row):
                                counts[i][value] += 1
                        return counts

                    def get_top_indices(counts, n, value):
                        indexed_counts = list(enumerate(counts))
                        sorted_counts = sorted(indexed_counts, key=lambda x: x[1][value], reverse=True)
                        return [index for index, _ in sorted_counts[:n]]
                    
                    counts = count_occurrences(data_layers)
                    remaining_indices = list(range(40))
                    results = []
                    selection_counts = [5, 5, 5, 5, 5, 5, 5, 5]   
                    
                    for value in range(8): 
                        n = selection_counts[value]
                        top_indices = get_top_indices([counts[i] for i in remaining_indices], n, value)
                        selected_indices = [remaining_indices[i] for i in top_indices]
                        results.append(selected_indices)
                        
                        if value < 7:  
                            remaining_indices = [i for i in remaining_indices if i not in selected_indices]
                    indices = torch.cat([torch.tensor(result).sort()[0] for result in results])
                    indices = torch.cat([torch.tensor(result).sort()[0] for result in results])
                    
                    data_all_layers_2.append(indices)
            manager.head_datas = torch.from_numpy(np.array(data_all_layers_2)).to(device)
        elif manager.method_name in manager.ahead_500_group32:
            data_all_layers_2 = []
            for i in range(manager.num_hidden_layers):
                numbers = 32
                if "llama-3" in manager.model_path:
                    logger.info("llama3")
                    filename = f"./search/llama3-instruct/32_groups/svd32/head_type_search_layer" + str(i) + ".csv"
                else:
                    filename = f"./search/llama2-chat/32_groups/svd{numbers}/head_type_search_layer" + str(i) + ".csv"
                data_layers = []
                if os.path.isfile(filename):
                    import csv
                    with open(filename, 'r', newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            data_layers.append([int(value) for value in row])
                else:
                    logger.error("load error")
                    raise ValueError
                def count_occurrences(arr):
                    counts = [[0 for _ in range(32)] for _ in range(32)] 
                    for row in arr:
                        for i, value in enumerate(row):
                            counts[i][value] += 1  
                    return counts

                def get_top_indices(counts, n, value):
                    indexed_counts = list(enumerate(counts))
                    sorted_counts = sorted(indexed_counts, key=lambda x: x[1][value], reverse=True)
                    return [index for index, _ in sorted_counts[:n]]

                counts = count_occurrences(data_layers)
                remaining_indices = list(range(32))
                results = []
                selection_counts = [1] * 32  
                
                for value in range(32): 
                    n = selection_counts[value]
                    top_indices = get_top_indices([counts[i] for i in remaining_indices], n, value)
                    selected_indices = [remaining_indices[i] for i in top_indices]
                    results.append(selected_indices)
                    
                    if value < 31:  
                        remaining_indices = [i for i in remaining_indices if i not in selected_indices]

                indices = torch.cat([torch.tensor(result).sort()[0] for result in results])

                data_all_layers_2.append(indices)
            manager.head_datas = torch.from_numpy(np.array(data_all_layers_2)).to(device)
        elif manager.method_name in manager.ahead_500_similar:
            data_all_layers_2 = []
            if "2" in args.method_name:
                select_topk = manager.num_attention_heads - 2
            elif "4" in args.method_name:
                select_topk = manager.num_attention_heads - 4
            elif "8" in args.method_name:
                select_topk = manager.num_attention_heads - 8
            else:
                select_topk = manager.num_attention_heads - 4
            manager.select_topk = select_topk
            logger.info(f"args.model_path is {args.model_path}")
            for i in range(manager.num_hidden_layers):
                filename = "./search/512/llama2-chat/query/head_type_search_layer" + str(i) + ".csv"
                data_layers = []
                if os.path.isfile(filename):
                    import csv
                    with open(filename, 'r', newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            data_layers.append([int(value) for value in row])
                else:
                    logger.error("load error")
                    raise ValueError
                data_layers = np.array(data_layers)
                data_layers = data_layers.sum(axis=0)
                num_heads = manager.num_attention_heads // 2
                top_half_indices = np.argpartition(data_layers, -manager.select_topk)[-manager.select_topk:]
                down_half_indices = np.argpartition(data_layers, -manager.select_topk)[:manager.num_attention_heads-manager.select_topk]
                indices = torch.cat([torch.tensor(down_half_indices).sort()[0],torch.tensor(top_half_indices).sort()[0]])
                data_all_layers_2.append(indices)
            manager.head_datas = torch.from_numpy(np.array(data_all_layers_2)).to(device)

    def add_lists(self,manager):
        self.init_manager(manager)
        self.rope_survey(manager)
        self.last_process(manager)
        self.not_update(manager)
        self.other_methods(manager)
        self.calib_sets(manager)
        
        self.uncomp_sets(manager)
        self.uncomp_extend(manager)
        self.multi_group_sets(manager)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="eager", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    
    parser.add_argument("--fphalf", type=int, default=1, help="If given, we will use fp16.")
    parser.add_argument("--method_name", type=str, default="", help="label for the output file.")
    parser.add_argument("--pattern", type=str, default="info", help="logging pattern.",choices=["debug", "info", "warning", "error", "critical"])
    args = parser.parse_args()
    patterns = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logger.set_console_level(patterns[args.pattern])

    set_seed(args.seed)
    logger.info(f"args.seed is {args.seed}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )
    model_path = args.model_path.lower()
    logger.info(f"model_path is {model_path} ")
    
    manager = Manager()
    manager.method_name = args.method_name
    manager.bsz = args.eval_batch_size
    manager.model_path = model_path
    
    func_utils_instance = func_utils()
    func_utils_instance.add_lists(manager)
    
    if args.fphalf:
        load_type = torch.float16
    else:
        load_type = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map={"": accelerator.process_index},
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation,
        token="",
    )
    accelerator.wait_for_everyone()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if "llama" in model_path.lower() or "mistral" in model_path.lower():
        from uncomp.monkeypatch import replace_llama,replace_mistral
        replace_llama(args.method.lower(),manager)
        replace_mistral(args.method.lower(),manager)
        func_utils_instance.determine_head_type(manager,device=model.device)
    
    model.eval()
    save_dir = args.save_dir   
    manager.calib_label = 0
    manager.func_utils_instance = func_utils_instance
    if manager.method_name in manager.calib:
        datas = get_calib_data("wikitext2", tokenizer, args.model_path, nsamples = 100,)
        manager.calib_datas = datas
        manager.calib_label = 1
        args.dataset = "narrativeqa"
        args.data_file = f"./data/LongBench/{args.dataset}.jsonl"
        # calib_main(args,manager)
        main(args,manager)
    else: 
        max_capacity_prompts = args.max_capacity_prompts
        for idx, dataset in enumerate(datasets):
            logger.info(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
            args.dataset = dataset
            args.data_file = f"./data/LongBench/{args.dataset}.jsonl"
            main(args,manager)