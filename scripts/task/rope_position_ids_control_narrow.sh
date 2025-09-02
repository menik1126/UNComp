count=0
gpu_id=0

echo "gpu_id is:"$gpu_id

for i in $(awk 'BEGIN{for(i=0.1;i<=0.9;i+=0.1)print i}'); do
    
    if (( $(echo "$i == 0.3" | bc -l) )); then
        echo "Skipping i=0.3"
        continue
    fi
    
    echo "i is:"$i

    nohup sh ./scripts/scripts_longBench/eval.sh \
    --max_capacity_prompts 512 \
    --attn_implementation eager \
    --source_path ./results/fp16/ \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --eval_batch_size 1 \
    --method rope_position_ids_control_narrow_$i \
    --name output/1012 \
    --gpu_id $gpu_id \
    --fp16 1 \
    --seed 43 \
    --logger_pattern info \
    --port 1234 > ./debug/1012/rope_position_ids_control_narrow_$i.log 2>&1 &

    count=$((count + 1))
    echo "count is:"$count
    
    if [ $((count % 2)) -eq 0 ]; then
        gpu_id=$((gpu_id + 1))
        echo "gpu_id is:"$gpu_id

        if [ $gpu_id -gt 7 ]; then
            echo "gpu_id 超过 7, 终止脚本执行。"
            break
        fi
    fi
done
