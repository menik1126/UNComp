import os
import json
import argparse
import numpy as np
from distutils.util import strtobool

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

from infinitebench_eval import (
    get_score_one_kv_retrieval,
    get_score_one_kv_retrieval,
    get_score_one_kv_retrieval,
    get_score_one_passkey,
    get_score_one_number_string,
    get_score_one_code_run,
    get_score_one_code_debug,
    get_score_one_longdialogue_qa_eng,
    get_score_one_longbook_qa_eng,
    get_score_one_longbook_sum_eng,
    get_score_one_longbook_choice_eng,
    get_score_one_longbook_qa_chn,
    get_score_one_math_find,
    get_score_one_math_calc,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "qasper_new": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_en_e": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "hotpotqa_new": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "2wikimqa_new": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "trec_new": classification_score,
    "triviaqa": qa_f1_score,
    "triviaqa_new": qa_f1_score,
    "samsum": rouge_score,
    "samsum_new": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_retrieval_en_new": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "repobench-p_new": code_sim_score,
    
    # Retrieve
    "kv_retrieval": get_score_one_kv_retrieval,
    "kv_retrieval_prefix": get_score_one_kv_retrieval,
    "kv_retrieval_both": get_score_one_kv_retrieval,

    "passkey": get_score_one_passkey,
    "number_string": get_score_one_number_string,
    # Code
    "code_run": get_score_one_code_run,
    "code_debug": get_score_one_code_debug,
    # Longbook
    "longdialogue_qa_eng": get_score_one_longdialogue_qa_eng,
    "longbook_qa_eng": get_score_one_longbook_qa_eng,
    "longbook_sum_eng": get_score_one_longbook_sum_eng,
    "longbook_choice_eng": get_score_one_longbook_choice_eng,
    "longbook_qa_chn": get_score_one_longbook_qa_chn,
    # Math
    "math_find": get_score_one_math_find,
    "math_calc": get_score_one_math_calc,
    #pg19
    "pg19": lambda *args: None,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--new_method', type=str, default=None)
    parser.add_argument('--switch', type=strtobool ,default=False, help="Switch to new method")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "trec_new", "triviaqa", "triviaqa_new", "samsum", "samsum_new", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def calc_score(dataset, prediction, ground_truths, all_classes):
    if dataset in ["code_debug"]:
        return get_score_one_code_debug(prediction, ground_truths)
    score = 0.
    # print("ground_truths", ground_truths)
    for ground_truth in ground_truths:
        # print("ground_truth", ground_truth)
        # print("type(ground_truth)", type(ground_truth))
        # print("begin begin")
        score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        # print("end end")
        
    return score

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        # print("begin")
        score = 0.
        if dataset in ["trec","trec_new", "triviaqa", "triviaqa_new","samsum", "samsum_new", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        # print("dataset", dataset)
        score = calc_score(dataset, prediction, ground_truths, all_classes)
        # for ground_truth in ground_truths:
        #     score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
        # print("end")
        # print("total_score: ", total_score)
        # print("len(predictions)",len(predictions))
        # print("round(100 * total_score / len(predictions), 2)",round(100 * total_score / len(predictions), 2))
    # print("end")
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    
    dataset_list = [
        "passkey", 
        "number_string",
        "kv_retrieval", 
        "longbook_qa_eng", 
        "longbook_choice_eng", 
        "longbook_sum_eng", 
        "longbook_qa_chn", 
        "math_find", 
        "math_calc", 
        "code_run", 
        "code_debug", 
        "longdialogue_qa_eng"
    ]
    
    print("args.switch", args.switch)
    if args.switch:
        print("true")
        new_result_list = []
        new_result_list.append([args.new_method])
    else:
        print("false")
        my_str = "rope_position_ids_control_narrow_dynamic_reverse"
        new_result_list = [
        ]
        for i in range(10,13,2):
            my_list = [my_str + "_" + str(i)]
            new_result_list.append(my_list)
        # for i in np.arange(0.1, 1.0, 0.1):
        #     my_list = [my_str + "_" + str(round(i, 1))]  # 使用 round 保留一位小数
        #     new_result_list.append(my_list)

        # print(new_result_list)
    
    results_list = new_result_list.copy()
    results_list.insert(0, ["dataset"])
    print("new_result_list", new_result_list)
    print("results_list", results_list)
    # if ["args.new_method"] not in results_list:
    #     results_list.append([args.new_method])
    
    for dataset in dataset_list:
        results_list[0].append(dataset)
        
        for idx, method in enumerate(new_result_list):
            try:
                method = method[0]
                args.method = method
                args.dataset = dataset
                # print("args.dataset", args.dataset)
                args.eval_file = os.path.join(args.results_dir,dataset,f"{method}.json")
                # print("args.eval_file", args.eval_file)
                scores = dict()
                predictions, answers, lengths = [], [], []
                with open(args.eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        # print(line)
                        try:
                            data = json.loads(line)
                            # print(data.keys())
                            predictions.append(data["pred"])
                            # print(data["pred"])
                            answers.append(data["answer"])
                            # print("answers", data["answer"])
                            all_classes = data["all_classes"]
                            if "length" in data:
                                lengths.append(data["length"])
                        except Exception as e:
                            print(f"Error occurred: {e}")
                            # print(f"Problematic line: {line}")
                            data = json.loads(line)
                            print(data.keys())
                            print("error error")
                            assert 1==0
                
                if args.longbench_e:
                    # print("longbench_e", score)
                    score = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                    # print("score", score)
                else:
                    # print("first in")
                    score = scorer(args.dataset, predictions, answers, all_classes)
                    if args.dataset == 'qasper' or args.dataset == 'qasper_new':
                        score_e = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                scores[args.dataset] = score
                    
                output_dir = os.path.dirname(args.eval_file)
                
                results_list[idx+1].append(score)
                
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
            
                print(f"dataset {args.dataset} method {args.method} scores {scores}")
            except:
                method = method[0]
                args.method = method
                args.dataset = dataset
                args.eval_file = os.path.join(args.results_dir,dataset,f"{method}.json")
                scores = dict()
                predictions, answers, lengths = [], [], []
                if os.path.exists(args.eval_file):
                    with open(args.eval_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                predictions.append(data["pred"])
                                answers.append(data["answer"])
                                all_classes = data["all_classes"]
                                if "length" in data:
                                    lengths.append(data["length"])
                            except:
                                print("error error error")
                                assert 1==0
                    if args.longbench_e:
                        score = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                    else:
                        score = scorer(args.dataset, predictions, answers, all_classes)
                        # print("scorer", score)
                        if args.dataset == 'qasper' or args.dataset == 'qasper_new':
                            score_e = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                    scores[args.dataset] = score
                        
                    output_dir = os.path.dirname(args.eval_file)
                    
                    results_list[idx+1].append(score)
                    
                    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                        json.dump(scores, f, ensure_ascii=False, indent=4)
                
                    print(f"dataset {args.dataset} method {args.method} scores {scores}")
                
    import csv
    with open(os.path.join(args.results_dir,f"results.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)
