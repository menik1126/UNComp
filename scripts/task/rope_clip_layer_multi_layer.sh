gpu_id=0
count=0
myport=1255
echo "gpu_id is:"$gpu_id

for i in $(seq 6 1 6); do
    myport=$((myport + 1)) 
    nohup sh ./scripts/scripts_longBench/eval.sh --max_capacity_prompts 512 --attn_implementation eager --source_path ./results/fp16/ --model_path meta-llama/Llama-2-7b-chat-hf --eval_batch_size 1 --method rope_clip_layer_multi_layer_$i --name ./output  --gpu_id _multi_2 --fp16 1 --seed 43 --logger_pattern info --port $myport > ./debug/1009/rope_clip_layer_multi_layer_$i.log 2>&1 &

    count=$((count + 1))
    echo "count is:"$count
    # if [ $((count % 2)) -eq 0 ]; then
        
    #     gpu_id=$((gpu_id + 1))
    #     echo "gpu_id is:"$gpu_id
    #     if [ $gpu_id -gt 7 ]; then
    #         echo "gpu_id 超过 7, 终止脚本执行。"
    #         break
    #     fi    
    # fi
done
