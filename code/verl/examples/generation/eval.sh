test_data_path=...
test_data_path=...
test_model_path=...

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$test_data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$test_model_path \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.max_num_batched_tokens=85535 \
    rollout.prompt_length=30000 \
    rollout.response_length=10000 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8

