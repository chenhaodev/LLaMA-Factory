#pip install -qqq runpod --progress-bar off

import runpod
import sys

MODEL = sys.argv[1] #"mlabonne/NeuralMarcoro14-7B" # @param {type:"string"}
GPU = sys.argv[2] #"NVIDIA GeForce RTX 3090" # @param ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA A30", "NVIDIA A40", "NVIDIA GeForce RTX 3070", "NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 3080 Ti", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090 Ti", "NVIDIA GeForce RTX 4070 Ti", "NVIDIA GeForce RTX 4080", "NVIDIA GeForce RTX 4090", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA L4", "NVIDIA L40", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX 4000 SFF Ada Generation", "NVIDIA RTX 5000 Ada Generation", "NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX A2000", "NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX A5000", "NVIDIA RTX A6000", "Tesla V100-FHHL-16GB", "Tesla V100-PCIE-16GB", "Tesla V100-SXM2-16GB", "Tesla V100-SXM2-32GB"]
NUMBER_OF_GPUS = sys.argv[3] # @param {type:"slider", min:1, max:8, step:1}
CONTAINER_DISK = 100 # @param {type:"slider", min:50, max:500, step:25}
VOLUME_IN_GB = 0
DB = 'oncc_medqa_instruct'
CLOUD_TYPE = "COMMUNITY" # @param ["COMMUNITY", "SECURE"]
REPO = "https://github.com/chenhaodev/LLaMA-Factory.git" # @param {type:"string"}
TRUST_REMOTE_CODE = True #False # @param {type:"boolean"}
DEBUG = False # @param {type:"boolean"}

# @markdown ---
RUNPOD_TOKEN = sys.argv[4] #"runpod" # @param {type:"string"}
GITHUB_TOKEN = sys.argv[5] #"github" # @param {type:"string"}
HF_TOKEN = sys.argv[6]
TRAIN_ARGS_EXT = sys.argv[7] 
RUNPOD_TEMPLATE = sys.argv[8] #when template_id='902i850eaw', download model, finetune, and upload; when template_id='0to5en2pon', only download model; e.g. bash -c 'cd /workspace/; git clone https://github.com/chenhaodev/LLaMA-Factory; cd /workspace/LLaMA-Factory/; sh auto-train.sh'; 

# Environment variables
runpod.api_key = RUNPOD_TOKEN
GITHUB_API_TOKEN = GITHUB_TOKEN
MODELRE = f"{MODEL.split('/')[-1]}-{DB}-v1"

# Create a pod
pod = runpod.create_pod(
    name=f"Finetune {MODEL.split('/')[-1]} on {DB.capitalize()}",
    image_name="runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_type_id=GPU,
    cloud_type=CLOUD_TYPE,
    gpu_count=NUMBER_OF_GPUS,
    volume_in_gb=VOLUME_IN_GB,
    container_disk_in_gb=CONTAINER_DISK,
    env={
        "MODEL": MODEL,
        "MODELRE": MODELRE,
        "REPO": REPO,
        "TRUST_REMOTE_CODE": TRUST_REMOTE_CODE,
        "DEBUG": DEBUG,
        "GITHUB_API_TOKEN": GITHUB_API_TOKEN,
        "HF_TOKEN": HF_TOKEN,
        "NUMBER_OF_GPUS": NUMBER_OF_GPUS,
        "TRAIN_ARGS_EXT": TRAIN_ARGS_EXT,
    },
    template_id=RUNPOD_TEMPLATE, 
)

print("Pod started: https://www.runpod.io/console/pods")
