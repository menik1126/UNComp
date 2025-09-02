gpu_id=2
count=0
echo "gpu_id is:"$gpu_id
#20
for i in $(seq 23 3 23); do
    nohup sh ./scripts/scripts_longBench/eval.sh --max_capacity_prompts 512 --attn_implementation eager --source_path ./results/fp16/ --model_path meta-llama/Llama-2-7b-chat-hf --eval_batch_size 1 --method rope_clip_layer_single_layer_$i --name ./output  --gpu_id $gpu_id --fp16 1 --seed 43 --logger_pattern info --port 1234 > ./debug/1009/rope_clip_layer_single_layer_$i.log 2>&1 &

    count=$((count + 1))
    echo "count is:"$count
    if [ $((count % 2)) -eq 0 ]; then
        
        gpu_id=$((gpu_id + 1))
        echo "gpu_id is:"$gpu_id

        if [ $gpu_id -gt 7 ]; then
            echo "gpu_id 超过 5, 终止脚本执行。"
            break
        fi
    fi
done
