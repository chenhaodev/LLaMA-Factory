#!/bin/bash
sh base-auto-train.sh 

start=$(date +%s)
# Detect the number of NVIDIA GPUs and create a device string
gpu_count=$(nvidia-smi -L | wc -l)
if [ $gpu_count -eq 0 ]
then
    echo "No NVIDIA GPUs detected. Exiting."
    exit 1
fi

# Install dependencies
apt update
apt install -y screen vim git-lfs

# Download basemodel
cd /workspace/; mkdir -p model model-update; cd LLaMA-Factory/; pip install -r requirements.txt; pip install bitsandbytes huggingface_hub hf_transfer; HF_HUB_ENABLE_HF_TRANSFER=1; 
huggingface-cli login --token $HF_TOKEN; huggingface-cli download $MODEL --local-dir /workspace/model 

# Run training. e.g. TRAIN_ARGS_EXT="--per_device_train_batch_size 4 --gradient_accumulation_steps 4 --lora_target q_proj,v_proj --template mistral --dataset oncc_medqa_instruct" check @ https://huggingface.co/spaces/hiyouga/LLaMA-Board

if [ $NUMBER_OF_GPUS -gt 1 ]
then
    cd /workspace/LLaMA-Factory; pip install tiktoken transformers_stream_generator; 
    sed -i "s=num_processes: 2=num_processes: $NUMBER_OF_GPUS=g" accelerate_config.yaml
    accelerate launch --config_file accelerate_config.yaml src/train_bash.py --stage sft --do_train True --model_name_or_path /workspace/model --finetuning_type lora --quantization_bit 4 --flash_attn True --dataset_dir data --cutoff_len 1024 --learning_rate 0.0005 --num_train_epochs 1.0 --max_samples 10000 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 100 --warmup_steps 20 --neftune_noise_alpha 0.5 --lora_rank 8 --lora_dropout 0.2 --output_dir /workspace/model-update $TRAIN_ARGS_EXT
else
    cd /workspace/LLaMA-Factory; 
    CUDA_VISIBLE_DEVICES=0 python src/train_bash.py --stage sft --do_train True --model_name_or_path /workspace/model --finetuning_type lora --quantization_bit 4 --flash_attn True --dataset_dir data --cutoff_len 1024 --learning_rate 0.0005 --num_train_epochs 1.0 --max_samples 10000 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 10 --save_steps 100 --warmup_steps 20 --neftune_noise_alpha 0.5 --lora_rank 8 --lora_dropout 0.2 --output_dir /workspace/model-update $TRAIN_ARGS_EXT
fi

cd /workspace/model-update/; sed -i "s=/workspace/model=$MODEL=g" README.md; sed -i "s=/workspace/model=$MODEL=g" adapter_config.json; huggingface-cli upload $MODELRE . . 

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds" 

sleep infinity
