#!/bin/bash

# Set the visible ROCm devices
method="" # Support "uncomp" "uncomp_stage" "uncomp_groupn"(n support 3 4 5 8, uncomp means default 2) "H2O" "snapkv" "pyramidkv_generate" "chai"
# Ultimate compressibility: uncomp_delete_head_n  uncomp_extreme_compressibility_n
# 
max_capacity_prompts="" # 512/86 in paper( (512+256)//2=384 / (86+43)//2=64 ) 
attn_implementation="" # Support "eager".
source_path="" # Result saving path
model_path="" # Model path
eval_batch_size="" # batch_size
data_path="" # debug file path
method_name=""    # method name
gpu_id=""       # accelerate profile
fp16=""         # Fp16 or not
seed=""         # seed
logger_pattern="" # logger pattern
port="" # port

while [ "$#" -gt 0 ]; do
  case "$1" in
    --method) method="$2"; shift 2;; 
    --max_capacity_prompts) max_capacity_prompts="$2"; shift 2;;
    --attn_implementation) attn_implementation="$2"; shift 2;;
    --source_path) source_path="$2"; shift 2;;
    --seed) seed="$2"; shift 2;;
    --model_path) model_path="$2"; shift 2;;
    --eval_batch_size) eval_batch_size="$2"; shift 2;;
    --name) name="$2"; shift 2;;
    --gpu_id) gpu_id="$2"; shift 2;;
    --fp16) fp16="$2"; shift 2;;
    --logger_pattern) logger_pattern="$2"; shift 2;;
    --port) port="$2"; shift 2;;
    --) shift; break;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

current_path_PWD=$PWD
echo "Current path: $current_path_PWD"
name="${current_path_PWD}/${name}/"
mkdir -p $name
echo "Output Catalogue" $name

method_name=$method

# Verify required arguments are provided
if [ -z "$method" ] || [ -z "$max_capacity_prompts" ] || [ -z "$attn_implementation" ] || [ -z "$source_path" ] || [ -z "$model_path" ]; then
  echo "Usage: $0 <ROCM_VISIBLE_DEVICES> --method <method> --max_capacity_prompts <max_capacity_prompts> --attn_implementation <attn_implementation> --source_path <source_path> --model_path <model_path>"
  exit 1
fi
gpu_id="/home/avnet/xiongjing/UNComp/scripts/scripts_longBench/gpu${gpu_id}.yaml"
save_dir="${source_path}results_long_bench"
accelerate launch --config_file $gpu_id --main_process_port $port run_longbench.py \
    --method ${method} \
    --seed ${seed} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True \
    --method_name ${method_name} \
    --fphalf $fp16 \
    --pattern $logger_pattern \
    --eval_batch_size ${eval_batch_size} > "${name}${method_name}_result.txt" \