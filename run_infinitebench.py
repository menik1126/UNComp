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
from pathlib import Path

# 创建logger实例

accelerator = Accelerator()
logger = Logger()

# datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
#              "qmsum",  "trec", "triviaqa", "samsum", \
#             "passage_count", "passage_retrieval_en", "lcc", "repobench-p","gov_report","multi_news",] 

datasets= ["passkey", "number_string", "kv_retrieval", "longbook_qa_eng", "longbook_choice_eng", 
            "longbook_qa_chn", "math_find",
           "code_run", "code_debug", "longdialogue_qa_eng"] #, "longbook_sum_eng","math_calc"

datasets= ["passkey",  "longbook_qa_chn", "math_find", "code_debug", "longdialogue_qa_eng" ,"longbook_choice_eng"]
datasets = ["number_string", "kv_retrieval","longbook_qa_eng", "code_run","longbook_sum_eng","math_calc"]
# datasets = ["longbook_qa_eng","longbook_choice_eng","longdialogue_qa_eng",
#            "longbook_qa_chn","code_debug","code_run","math_find",
#            "passkey","number_string","kv_retrieval"]

# datasets = ["narrativeqa", "musique","qmsum",  "trec", "passage_count",  "lcc"] 

# datasets = ["lcc", "repobench-p"] 

# datasets = ["gov_report","multi_news"]

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
    "repobench-p_new": 64 ,
    
    # infinitebench
    "passkey": 32,
    "number_string": 32,
    "kv_retrieval": 100, 
    "longbook_qa_eng": 32, 
    "longbook_choice_eng": 32,
    "longbook_sum_eng": 1536,
    "longbook_qa_chn": 32,
    "math_find": 32,
    "math_calc": 64000,
    "code_run": 32,
    "code_debug": 32, 
    "longdialogue_qa_eng": 32,
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

    # infinitebench
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    # "longbook_sum_eng": "Summarize the book below:\n\n{context}",  # noqa
    "longbook_qa_eng": "Read the book below and answer a question.\n\n{context}\n\nQuestion: {question}\n\nBe very concise.",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}",  # noqa
    "longbook_sum_eng": "Summarize the following book.\n\n{context}",  # noqa
    "longbook_qa_chn": "请根据以下书籍回答我的问题。\n\n{context}\n\n问题：{question}\n请尽量简短地回答。",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Compute the intermediate values in the following long expression.\n\n{context}",  # noqa
    "code_run": "Following is a set of Python functions. There is a function called named {func}.\n\n{context}\n\nPlease give me the exact number of the return value of {func_call}. Be concise. Your response must end with the final returned value.",  # noqa
    "code_debug": "There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion has deliberate error?\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nYou should first find the functions in the options. Repeat their content, inspect through code, and at last give me your answer for the function that has the deliberate and obvious error in A, B, C, or D.",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else.",  # noqa
}

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 3950,
    "longlora": 12000,
    # "mistral": 12000,
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
    prompts_all = []
    inputs = []
    contexts = []
    answerss = []
    ids = []
    datasets = []
    optionss = []
    input_max_len = 0
    all_classess = []
    lengths = []
    
    model_path = args.model_path.lower()
    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
 
    output_max_len = dataset2maxlen[args.dataset]

    DATA_NAME_TO_PATH = {
        # Retrieval tasks
        "passkey": "passkey.jsonl",
        "number_string": "number_string.jsonl",
        "kv_retrieval": "kv_retrieval.jsonl",
        # Book tasks
        "longbook_sum_eng": "longbook_sum_eng.jsonl",
        "longbook_choice_eng": "longbook_choice_eng.jsonl",
        "longbook_qa_eng": "longbook_qa_eng.jsonl",
        "longbook_qa_chn": "longbook_qa_chn.jsonl",
        # "book_qa_eng": "longbook_eng/longbook_qa_eng.jsonl",
        "longdialogue_qa_eng": "longdialogue_qa_eng.jsonl",
        # Math tasks
        "math_find": "math_find.jsonl",
        "math_calc": "math_calc.jsonl",
        # Code tasks
        "code_run": "code_run.jsonl",
        "code_debug": "code_debug.jsonl",
    }

    def iter_jsonl(fname, cnt=None):
        i = 0
        with open(fname, "r") as fin:
            for line in fin:
                if i == cnt:
                    break
                yield json.loads(line)
                i += 1
    
    def load_data(data_name: str, data_dir: str = "/home/avnet/xiongjing/UNComp/data/InfiniteBench/"):
        path = DATA_NAME_TO_PATH[data_name]
        fname = Path(data_dir, path)
        return list(iter_jsonl(fname))
    
    examples = load_data(data_name=args.dataset)
    logger.info(f"type(examples): {type(examples)}")
    logger.info(f"len(examples): {len(examples)}")
    logger.info(f"type(examples[0]): {examples[4]['answer']}")
    logger.info(f"examples[0].keys(): {examples[0].keys()}")
    logger.info("Finish loading InfiniteBench")
    
    def get_answer(eg: dict, data_name: str):
        if data_name in ["code_debug", "longbook_choice_eng"]:
            OPTIONS = "ABCD"
            if isinstance(eg["answer"], str):
                ret = [eg["answer"], OPTIONS[eg['options'].index(eg["answer"])]]
            elif isinstance(eg["answer"], list):
                if len(eg["answer"]) == 1:
                    ret = [eg["answer"][0], OPTIONS[eg['options'].index(eg["answer"][0])]]
                elif len(eg["answer"]) == 2 and eg["answer"][1] in ['A', 'B', 'C', 'D']:
                    ret = eg['answer']
                else:
                    raise ValueError
            else:
                raise ValueError
            return ret

        return eg["answer"]
    
    def create_prompt(eg,data_name):
        template = model2prompt[data_name]
        if data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            return template.format(
                func=func,
                func_call=func_call,
                context=eg["context"],
            )
        elif data_name in ["code_debug", "code_debug_qa"]:
            code = eg["context"]
            if data_name == "code_debug":
                return template.format(
                    context=code,
                    OPTION_A=eg["options"][0],
                    OPTION_B=eg["options"][1],
                    OPTION_C=eg["options"][2],
                    OPTION_D=eg["options"][3],
                )
            return template.format(context=code)
        elif data_name == "longdialogue_qa_eng":
            script = eg["context"]
            prompt = template.format(context=script)
            return prompt
        elif data_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            book = eg["context"]
            if data_name == "longbook_choice_eng":
                return template.format(
                    question=eg["input"],
                    context=book,
                    OPTION_A=eg["options"][0],
                    OPTION_B=eg["options"][1],
                    OPTION_C=eg["options"][2],
                    OPTION_D=eg["options"][3],
                )
            elif data_name == "longbook_qa_eng":
                return template.format(
                    question=eg["input"],
                    context=book,
                )
            elif data_name == "longbook_sum_eng":
                return template.format(context=book)
            elif data_name == "longbook_qa_chn":
                return template.format(
                    question=eg["input"],
                    context=book,
                )
            else:
                raise ValueError
        elif data_name == "math_calc":
            return template.format(context=eg["context"])
        elif data_name == "math_find":
            prompt = eg['input']
            context = eg['context']
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            prefix = f"What is {target_number} in the following list?"
            return template.format(
                prefix=prefix,
                context=context,
                input=prompt,
            )

        # Default behavior if content key exists
        if "content" in eg:
            content = eg["content"]
            del eg["content"]
            eg["context"] = content

        format_dict = {
            "context": eg["context"],
            "input": eg["input"],
        }
        prompt = template.format(**format_dict)
    
    
        return prompt

    
    for example in examples:
        prompt = model2prompt[args.dataset]
        if "llama2" in args.model_path.lower():
            prompt = build_chat(prompt)
        example["prompt"] = create_prompt(example, args.dataset)
        example["length"] = len(example["context"].split())
        example["all_classes"] = None

    for example in examples:
        prompts_all.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(get_answer(example, args.dataset))
        print("get_answer(example, args.dataset)",get_answer(example, args.dataset))
        optionss.append(example["options"])
        datasets.append(args.dataset)
        all_classess.append(example["all_classes"])
        lengths.append(example["length"])
        ids.append(example["id"])

    combined_data = [
        {
            "prompt": p, "input": i, "context": c, "answer": a, 
            "options": op, "id": id, "dataset": dataset,
        }
        for p, i, c, a, op, id, dataset in zip(
            prompts_all, inputs, contexts, answerss, optionss, ids,datasets
        )
    ]
    
    if manager.method_name in manager.rope_correlation:
        combined_data = combined_data[:1]
        output_max_len = 1
        manager.rope_correlation_dict["dataname"] = args.dataset
    # elif manager.method_name in manager.rope_position_ids_control:
    #     if "elongate" in manager.method_name:
    #         model_max_len = model_max_len//manager.rope_position_ids_control_dict["elongate_ratio"]
    
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
            batch_answerss = [item["answer"] for item in batch_data]
            batch_datasets = [item["dataset"] for item in batch_data]
            batch_options = [item["options"] for item in batch_data]
            batch_ids = [item["id"] for item in batch_data]
            
            tokenized_prompts = tokenizer(batch_prompts, 
                                          padding="longest", 
                                          return_tensors="pt", 
                                          add_special_tokens=True,
                                          ).to('cuda')
            # print("batch_prompts is ", batch_prompts)
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask
            actual_lengths = attention_mask.sum(dim=1)
            max_len = actual_lengths.max().item()
            padding_len = max_len - actual_lengths
            if args.eval_batch_size == 1:
                if len(batch_input_ids[0]) > model_max_len:
                    half = int(model_max_len/2)
                    # prompt = [tokenizer.decode(batch_input_ids[i][padding_len[i]:padding_len[i]+half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[i][-half:], skip_special_tokens=True) for i in range(len(batch_input_ids))]
                    # print("type(batch_input_ids) is ", batch_input_ids.shape)
                    prompt = tokenizer.decode(torch.cat([batch_input_ids[0,:half],batch_input_ids[0,-half:]]), skip_special_tokens=True)
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
            if manager.method_name in manager.draw_picture_set:
                output_max_len = 1
                manager.dataset = args.dataset
            
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
            batch_outputs =tokenizer.batch_decode(output[:,context_length:], skip_special_tokens=True)
            print("\nbatch_outputs is ", batch_outputs)
            # assert 1==0
            batch_generations = batch_outputs
            for j in range(len(batch_prompts)):
                example = {}
                example["prompt"] = batch_prompts[j]
                example["input"] = batch_inputs[j]
                example["context"] = batch_contexts[j]
                example["answer"] = batch_answerss[j]
                example["pred"] = batch_generations[j]
                example["options"] = batch_options[j]
                example["all_classes"] = all_classess[j]
                example["length"] = lengths[j]
                
                example["dataset"] = batch_datasets[j]
                example["id"] = batch_ids[j]
                results["outputs"].append(example)
                results["num_tokens"] += len(batch_generations[j])
            if manager.method_name in manager.draw_picture_set:
                break  
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
                        # print(f"GPU 2 VRAM Total Memory : {value :.2f} MB")
                        return value
            
            used = get_rocm_memory()
            if manager.max_used < used:
                manager.max_used = used
            if manager.min_used > used:
                manager.min_used = used
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
        # print(f"attn module sec/token: {(prefill_time_gathered + generate_time_gatherd ) / num_tokens}")
        
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
        # traindataset.append(tot_text[i:j])
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
        # last_process处理
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
            # 使用 re.findall 来匹配所有的整数和小数
            matches = re.findall(r'\d+(?:\.\d+)?', string)
            if matches:
                numbers = [float(num) for num in matches]  # 将匹配到的数字转换为浮点数列表
                print(f"All numbers in the string are: {numbers}")
            else:
                print("No numbers found in the string.")
            return numbers
        
        def get_last_float_number(string):
            # 修改正则表达式，匹配整数或小数
            match = re.search(r'(\d+(\.\d+)?)(?!.*\d)', string)
            if match:
                last_number = float(match.group(1))  # 将结果转换为浮点数
                print(f"The last number in the string is: {last_number}")
            else:
                print("No number found in the string.")
            return last_number

        
        if "rope" in manager.method_name:
            manager.rope.append(manager.method_name)
            logger.info("rope")
            if "rope_survey" in manager.method_name:
                manager.rope_survey.append(manager.method_name)
                manager.rope_survey_dict = {
                    "hidden_states":[],
                    "query_states":[],
                    "key_states":[],
                    "value_states":[],
                    "query_states_rope":[],
                    "key_states_rope":[],
                }
                current_working_directory = os.getcwd()
                if "svd" in manager.method_name:
                    filename = current_working_directory+"/save_files/rope_svd_files/"
                else:
                    filename = current_working_directory+"/save_files/rope_files/"
                if os.path.exists(filename):    
                    logger.info(f"Directory {filename} already exists")
                else:
                    os.mkdir(filename)
                    logger.info(f"mkdir {filename}")
                
                manager.rope_survey_head_dict = {
                    "save_path": filename,
                    "hidden_states":[],
                    "query_states_head":[],
                    "key_states_head":[],
                    "value_states_head":[],
                    "query_states_head_rope":[],
                    "key_states_head_rope":[],
                    
                    "query_states_svd32_head_rope":[],
                    "key_states_svd32_head_rope":[],
                    "query_states_svd16_head_rope":[],
                    "key_states_svd16_head_rope":[],
                    "query_states_svd64_head_rope":[],
                    "key_states_svd64_head_rope":[],
                    "query_states_svd96_head_rope":[],
                    "key_states_svd96_head_rope":[],
                }
            elif "rope_clip_layer_single_layer" in manager.method_name:
                manager.rope_clip_layer_single_layer.append(manager.method_name)
                last_number = get_last_number(manager.method_name)
                logger.info(f"cut rope in layer {last_number}")
                manager.rope_clip_layer_single_layer_dict ={
                    "clip_layer": [last_number],
                }
            elif "rope_clip_layer_multi_layer" in manager.method_name:
                manager.rope_clip_layer_multi_layer.append(manager.method_name)
                last_number = get_last_number(manager.method_name)
                group_to_clip = {1:[24,25,26,27,28,29,30,31],2:[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
                                 3:[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
                                4:[28,29,30,31],5:[20,21,22,23,24,25,26,27,28,29,30,31],
                                6:[28,30],7:[10,20,30],8:[20,25,30],9:[20,22,24,26,28,30],
                }
                logger.info(f"cut rope in in layer {group_to_clip[last_number]}")
                manager.rope_clip_layer_multi_layer_dict = {
                    "clip_layers": group_to_clip[last_number],
                }
            elif "rope_correlation" in manager.method_name:
                logger.info("rope_correlation")
                manager.rope_correlation.append(manager.method_name)
                manager.rope_correlation_dict = {
                    "query_before_rope": [],
                    "query_after_rope": [],
                    "key_before_rope": [],
                    "key_after_rope": [],
                }
                logger.info(f"manager.rope_correlation_dict {manager.rope_correlation_dict}")
            elif "rope_position_ids_control" in manager.method_name:
                logger.info("rope_position_ids_control")
                manager.rope_position_ids_control.append(manager.method_name)
                if "v_dynamic" in manager.method_name:
                    layer = get_last_number(manager.method_name)
                    logger.info("rope_position_ids_control_v_dynamic")
                    interval = 1/layer * 2
                    manager.rope_position_ids_control_dict = {
                        "layer":layer,
                    }
                    for i in range(layer//2): #倒着看
                        manager.rope_position_ids_control_dict[manager.num_hidden_layers-i-1] = 1-interval*i
                    for i in range(layer//2,layer): #后加
                        logger.info(f"layer//2 {layer//2}")
                        manager.rope_position_ids_control_dict[manager.num_hidden_layers-i-1] = interval+interval*(i-layer//2)
                    logger.info(f"rope_position_ids_control_dict {manager.rope_position_ids_control_dict}")        
                     
                elif "v_group_dynamic" in manager.method_name:
                    numbers = get_all_numbers(manager.method_name)
                    layer = int(numbers[0]) # 后几层
                    groups_item_num = int(numbers[1]) #每组数量
                    min_ratio=numbers[2] #最低压缩比率
                    if len(numbers) != 3:
                        raise ValueError("min_ratio should be a float number")
                    groups_amount = layer//groups_item_num
                    logger.info("rope_position_ids_control_v_dynamic")
                    
                    interval = (1-min_ratio)/(groups_amount//2-1)
                    manager.rope_position_ids_control_dict = {
                        "layer":layer,
                    }
                    for i in range(groups_amount//2):
                        for j in range(groups_item_num):
                            manager.rope_position_ids_control_dict[manager.num_hidden_layers-i*groups_item_num-j-1] = 1 - interval*i
                    for i in range(groups_amount//2,groups_amount):
                        for j in range(groups_item_num):
                            manager.rope_position_ids_control_dict[manager.num_hidden_layers-i*groups_item_num-j-1] = min_ratio+interval*(i-groups_amount//2)
                    for j in range(groups_amount*groups_item_num,layer):
                        manager.rope_position_ids_control_dict[manager.num_hidden_layers-j-1] = 1
                    logger.info(f"rope_position_ids_control_dict {manager.rope_position_ids_control_dict}")        
                    # assert 1==0         
                elif "dynamic" in manager.method_name:
                    layer = get_last_number(manager.method_name)
                    logger.info("rope_position_ids_control_dynamic")
                    interval = 1/layer
                    manager.rope_position_ids_control_dict = {
                        "layer":layer,
                    }
                    if "reverse" in manager.method_name:
                        for i in range(layer):
                            manager.rope_position_ids_control_dict[manager.num_hidden_layers-i-1] = interval+interval*i
                        logger.info(f"rope_position_ids_control_dict {manager.rope_position_ids_control_dict}")
                    else:
                        for i in range(layer):
                            manager.rope_position_ids_control_dict[manager.num_hidden_layers-i-1] = 1-interval*i
                        logger.info(f"rope_position_ids_control_dict {manager.rope_position_ids_control_dict}")
                else:
                    ratio = get_last_float_number(manager.method_name)
                    logger.info(f"rope_position_ids_control_dict {ratio}")
                    manager.rope_position_ids_control_dict = {
                        "layer":4,
                        "elongate_ratio": ratio,
                        "narrow_ratio": ratio,
                    }
            
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
        manager.calib = ["head_type_search_2","head_type_search_3","head_type_search_4","head_type_search_5",
                     "head_type_search_8","head_type_search_2_variance"]
        if "calib" in manager.method_name:
            manager.calib.append(manager.method_name)

    def draw_picture_sets(self,manager):
        manager.draw_picture_set = ["draw_svd_trend","draw_thermodynamic_chart","draw_accumulated_energy","draw_entropy",
                                "draw_effective_rank"]
        if any(item in manager.method_name for item in manager.draw_picture_set):
            manager.draw_picture_set.append(manager.method_name)
      
    
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
        if "uncomp_stage" in manager.method_name:
            logger.info("uncomp_stage")
            manager.ahead_500.append(manager.method_name)
            manager.hidden_delete_stage_and_ours.append(manager.method_name)
            manager.ahead_500_equal_code.append(manager.method_name)
        
        
        if "uncomp" in manager.method_name and "extreme_compressibility" not in manager.method_name \
        and "group" not in manager.method_name and "delete_head" not in manager.method_name \
            and "pyramidkv" not in manager.method_name:
            logger.info("uncomp")
            manager.ahead_500_equal_code.append(manager.method_name)
            manager.ahead_500.append(manager.method_name)
        
        manager.pyramidkv_uncomp = []
        if "pyramidkv" in manager.method_name and "uncomp" in manager.method_name:
            logger.info("pyramidkv_uncomp")
            manager.ahead_500.append(manager.method_name)
            manager.pyramidkv_uncomp.append(manager.method_name)
        
        lists_to_extend = [manager.extreme_compressibility_equal_code,
                           manager.delete_head_equal_code,
                           manager.ahead_500_equal_code,
                           manager.pyramidkv_uncomp,
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
        if "uncomp_group3" in manager.method_name:
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
        
        manager.ahead_500.extend(manager.extreme_compressibility_equal_code)
        manager.ahead_500_similar.extend(manager.delete_head_equal_code)
        manager.delet_head_set.extend(manager.delete_head_equal_code)
    
    def determine_head_type(self,manager,device):
        if manager.method_name in manager.ahead_500:
            data_all_layers = []
            data_all_layers_2 = []
            logger.info(f"args.model_path is {args.model_path}")
            for i in range(manager.num_hidden_layers):
                if "variance" in manager.method_name:
                    filename = "./search/512/llama2-chat/variance/head_type_search_layer" + str(i) + ".csv"
                    logger.info(f"filename is {filename}")
                elif "uncomp" in manager.method_name:
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
                numbers = 32
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
            for i in range(manager.num_hidden_layers):
                numbers = 32
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
        self.draw_picture_sets(manager)
        
        self.uncomp_sets(manager)
        self.uncomp_extend(manager)
        self.multi_group_sets(manager)
        # self.determine_head_type(manager)

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
        token="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",
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
    if manager.method_name in manager.calib:
        datas = get_calib_data("wikitext2", tokenizer, args.model_path, nsamples = 500,)
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
            args.data_file = f"./data/InfiniteBench/{args.dataset}.jsonl"
            # args.data_file = f"./data/LongBench/{args.dataset}.jsonl"
            main(args,manager)