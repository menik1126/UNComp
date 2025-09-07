# UNCOMP: Can Information Compression Uncover Sparsity? — A Compressor Design from an Uncertainty-Aware Perspective

[![arXiv](https://img.shields.io/badge/arXiv-2410.03090-b31b1b.svg)](https://arxiv.org/abs/2410.03090)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official implementation of **UNCOMP**, an uncertainty-aware KV cache compression framework for long-context LLMs. It leverages truncated matrix entropy to reveal sparsity, cutting cache size to 4.74%, boosting throughput by 6×, with minimal performance loss.

## 📄 Paper

**UNCOMP: Can Information Compression Uncover Sparsity? — A
Compressor Design from an Uncertainty-Aware Perspective**  
*Jing Xiong, Jianghan Shen, Fanghua Ye, Chaofan Tao, Zhongwei Wan, Jianqiao Lu, Xun Wu, Chuanyang Zheng, Zhijiang Guo, Min Yang, Lingpeng Kong, Ngai Wong*

📖 [Paper Link](https://arxiv.org/abs/2410.03090)

## 🚀 Key Features
- **Uncertainty-Aware Compression**: Uses truncated matrix entropy to detect low-information regions
- **Two-Stage Framework**: Jointly compresses hidden states and KV cache, accelerating both prefill and decoding.
- **Near-Lossless Accuracy**: Outperforms or matches full KV cache in benchmarks like Needle-in-a-Haystack, even at 9.38% compression ratio.
- **Extreme Compression**: Reduces KV cache size to 4.74% of the original. Maintains high accuracy even at aggressive ratios (e.g., pruning multiple heads).
- **Efficiency**: Achieves 6× throughput improvement and 6% prefill speedup.
- **Training-free**: No costly fine-tuning required.

## 🛠️ Installation

### Requirements

- Python 3.9+
- PyTorch 2.6.0
- CUDA compatible GPU(s)

### Prerequisites & Dependencies

**Setup**

1. Clone the repository:
```bash
git clone https://github.com/your-username/UNComp.git
cd UNComp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
**Key Dependencies:**
- `torch==2.6.0` - PyTorch for deep learning
- `transformers==4.39.2` - Hugging Face transformers library
- `accelerate==1.0.1` - For multi-GPU training and inference
- `datasets==3.3.1` - For dataset loading and processing
- `numpy`, `pandas` - Data manipulation
- `sentencepiece==0.2.0` - For tokenization


### Quick Start

#### LongBench

```bash
sh ./scripts/scripts_longBench/eval.sh \
    --max_capacity_prompts 512 \
    --attn_implementation eager \
    --source_path ./results/ \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --eval_batch_size 1 \
    --method uncomp \
    --name ./output \
    --gpu_id 0 \
    --fp16 1 \
    --seed 43 \
    --logger_pattern info \
    --port 1236
```

### Supported Models

- **LLaMA** family models  
- **mistral** models

### Supported Datasets

#### Long Context Benchmarks
- **LongBench**: narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum, multi_news, trec, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p
- **InfiniteBench**: En.Sum, En.QA, En.MC, En.Dia, Zh.QA, Code.Debug, Code.Run, Math.Calc, Math.Find, Retrieve.PassKey, Retrieve.Number, Retrieve.KV
- **Needle in a Haystack Task**: A simple 'needle in a haystack' analysis to test in-context retrieval ability of long context LLMs.
- **Standard Benchmarks**: GSM8K

## 📈 Evaluation Scripts

### Preparation Stage
#### Layer Groups
```bash
python uncomp/stage_division.py
```
#### Head Groups
```bash
sh ./scripts/scripts_longBench/eval.sh --max_capacity_prompts 512 --attn_implementation eager --source_path ./results/ --model_path meta-llama/Llama-2-7b-chat-hf --eval_batch_size 1 --method head_type_search_2 --name ./output --gpu_id multi_0 --fp16 1 --seed 43 --logger_pattern info --port 1236
```

### LongBench Evaluation

Evaluate on LongBench datasets using the provided scripts:

```bash
# Generation 
### Multi-GPU evaluation 
sh ./scripts/scripts_longBench/eval.sh --max_capacity_prompts 512 --attn_implementation eager --source_path ./results/ --model_path meta-llama/Llama-2-7b-chat-hf --eval_batch_size 1 --method uncomp --name ./output --gpu_id multi_0 --fp16 1 --seed 43 --logger_pattern info --port 1236
### Single GPU evaluation  
sh ./scripts/scripts_longBench/eval.sh --max_capacity_prompts 512 --attn_implementation eager --source_path ./results/ --model_path meta-llama/Llama-2-7b-chat-hf --eval_batch_size 1 --method uncomp --name ./output --gpu_id 0 --fp16 1 --seed 43 --logger_pattern info --port 1236

# Evaluation
sh ./scripts/scripts_longBench/metrics.sh --results_dir  ./results/results_long_bench/llama-2-7b-chat-hf_512/ --switch True  --new_method uncomp
```

### Infinitebench Evaluation

Evaluate on LongBench datasets using the provided scripts:

```bash
# Generation
### Multi-GPU evaluation 
sh ./scripts/scripts_InfiniteBench/eval.sh --max_capacity_prompts 512 --attn_implementation eager --source_path ./results/ --model_path meta-llama/Llama-2-7b-chat-hf --eval_batch_size 1 --method uncomp --name ./output --gpu_id multi_0 --fp16 1 --seed 43 --logger_pattern info --port 1236
### Single GPU evaluation  
sh ./scripts/scripts_InfiniteBench/eval.sh --max_capacity_prompts 512 --attn_implementation eager --source_path ./results/ --model_path meta-llama/Llama-2-7b-chat-hf --eval_batch_size 1 --method uncomp --name ./output --gpu_id 0 --fp16 1 --seed 43 --logger_pattern info --port 1236

# Evaluation
sh ./scripts/scripts_InfiniteBench/metrics.sh --results_dir  ./results/results_Inifite_bench/llama-2-7b-chat-hf_512/ --switch True  --new_method uncomp
```

**Hyperparameter selection**: 
- method:
    - head_type_search_2: The heads are divided into two groups.
    - head_type_search_4: The heads are divided into four groups.
    - head_type_search_8: The heads are divided into eight groups.
    - head_type_search_32: The heads are divided into thirty-two groups.
    - uncomp: The heads are divided into two groups.
    - uncomp_stage: The heads are divided into two groups and layer are divided into some groups.
    - uncomp_groupn: The heads are divided into n groups.
    - other methods: snapkv/pyramidkv/fullkv/streamingllm/h2o/chai.
- max_capacity_prompts:
    - 512/128: The average length retained per head.


## 🚧 TODO & Roadmap
- [❎] **Code Organization**: Currently organizing and cleaning up the codebase for better usability
- [❎] **Qwen Support**: Adding full support for Qwen model family
- [❎] **Baselines**: Adding full support for Evaluation of Baselines
- [❎] **SGLang Integration**: Adding support for SGLang inference engine for improved performance
- [❎] **Documentation**: Expanding documentation with more detailed examples
- [❎] **Quantization Support**: Adding support for model quantization (INT8/INT4) to reduce memory usage and accelerate inference
- [❎] **Benchmarks**: Adding more comprehensive benchmark results
- [✅] **Multi-GPU Inference Support**
- [✅] **Batch Inference Support**
- [✅] **AMD GPU Support**

## 📁 Project Structure

```
UNComp/
├── modeling_*.py              # Model implementations with PCW
├── run_evaluation*.py         # Evaluation scripts
├── experiment_manager*.py     # Experiment management
├── pcw_wrapper*.py           # Parallel Context Window wrapper
├── datasets_loader.py        # Dataset loading utilities
├── eval_longbench.py         # LongBench evaluation
├── metrics.py                # Evaluation metrics
├── scripts/                  # Bash scripts and configs
├── longbench_config/         # LongBench configurations
├── results/                  # Output results
└── requirements.txt          # Dependencies
```

## 📊 Results

ParallelComp achieves significant improvements in long-context tasks:

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{xiong2025parallelcomp,
  title={ParallelComp: Parallel Long-Context Compressor for Length Extrapolation},
  author={Xiong, Jing and Shen, Jianghan and Zheng, Chuanyang and Wan, Zhongwei and Zhao, Chenyang and Yang, Chiwun and Ye, Fanghua and Yang, Hongxia and Kong, Lingpeng and Wong, Ngai},
  journal={arXiv preprint arXiv:2502.14317},
  year={2025}
}
```



## 📞 Contact

For questions and support, please open an issue in this repository or contact the authors.

---

**Note**: This implementation will be fully released soon. Stay tuned for updates!
