task_name='rank_32'

accelerate launch --config_file "/root/.cache/huggingface/accelerate/default_config.yaml" finetune.py \
  --base_model 'meta-llama/Llama-3.1-8B-Instruct' \
  --output_dir './trained_models/'${task_name} \
  --batch_size 32 \
  --micro_batch_size 2 \
  --num_epochs 3 \
  --learning_rate 1e-5 \
  --cutoff_len 1024 \
  --val_set_size 0 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_target_modules "['q_proj', 'k_proj', 'v_proj', 'down_proj', 'up_proj']" > ./logs/${task_name}.log 2>&1