$HOME=...

python legacy_model_merger.py merge \
    --backend megatron \
    --tie-word-embedding \
    --hf_model_path $HOME/checkpoints/Llama-3.2-1B-Instruct \
    --local_dir $HOME/code/verl/examples/grpo_trainer/checkpoints/llama_exp/skywork/global_step_20/actor \
    --target_dir $HOME/code/verl/examples/grpo_trainer/checkpoints/llama_exp/skywork/global_step_20/actor_merged_hf

