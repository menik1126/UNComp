# This script is adapted from 
# https://github.com/FranxYao/Long-Context-Data-Engineering.git


mkdir -p ./results_needle/logs/
mkdir -p ./results_needle/img/


METHOD='uncomp'       # ['uncomp'','fullkv', 'pyramidkv_generate', 'snapkv', 'chai', 'H2O', 'uncomp_stage']
MAX_CAPACITY_PROMPT=512  # [86, 512 ...]
attn_implementation="eager" # Support "eager".
TAG=test


# For Llama3-8b
# (
# python -u run_needle_in_haystack.py --s_len 7000 --e_len 8000\
#     --model_provider LLaMA3 \
#     --model_name "/home/avnet/xiongjing/PyramidKV/pyramidkv/meta-llama/Meta-Llama-3-8B-Instruct" \
#     --attn_implementation ${attn_implementation} \
#     --step 100 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# For Llama2-7b
(
python -u run_needle_in_haystack.py --s_len 3000 --e_len 4000\
    --model_provider LLaMA \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --attn_implementation ${attn_implementation} \
    --step 100 \
    --method $METHOD \
    --max_capacity_prompt $MAX_CAPACITY_PROMPT \
    --model_version LlaMA2_new_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
) 2>&1  | tee results_needle/logs/LlaMA2_new_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log


# For Mistral

# (
# python -u run_needle_in_haystack.py --s_len 400 --e_len 32001\
#     --model_provider Mistral \
#     --model_name YOU_PATH_TO_MISTRAL_2 \
#     --step 400 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Mistral2_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee logs/Mistral2_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log