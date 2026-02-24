vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name sharegpt \
  --dataset-path /data/maulik/shareGPT4V/Image/sharegpt4v_instruct_gpt4-vision_cap100k.json \
  --num-prompts 100 \
  --save-result \
  --result-dir ~/vllm_benchmark_results \
  --save-detailed \
  --endpoint /v1/chat/completions


  # run benchmarking script
vllm bench serve --save-result --save-detailed \
  --backend openai-chat \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name custom \
  --dataset-path sample.jsonl \
  --allowed-local-media-path /data/maulik/shareGPT4V/Image/


vllm bench serve --save-result --save-detailed \
  --backend openai-chat \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name custom \
  --dataset-path sample_v.jsonl 



vllm bench serve --save-result --save-detailed \
  --backend openai-chat \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name custom \
  --dataset-path sample.jsonl 


CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --allowed-local-media-path /data/maulik/shareGPT4V/Image/
