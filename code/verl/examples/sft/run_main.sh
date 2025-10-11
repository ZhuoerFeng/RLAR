# Tested with 2 & 4 GPUs

# data.train_files=$HOME/data/huggingRM/verlsft/dataset/english-french-translation_en-fr.parquet \

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_peft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

export HOME=...
export WANDB_API_KEY=...

nproc_per_node=$1
save_path=$2

shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/valid.parquet \
    data.max_length=30000 \
    data.truncation=left \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-5 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=1 \
    data.micro_batch_size_per_gpu=1 \
    data.val_batch_size=1 \
    model.partial_pretrain=$HOME/checkpoints/Llama-3.2-1B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=main_exp \
    trainer.experiment_name=sft-llama-3.2-1b \
    trainer.logger=['console'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
