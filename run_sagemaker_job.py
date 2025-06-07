import os
from dotenv import load_dotenv
import sagemaker
from sagemaker.pytorch import PyTorch

if not load_dotenv(".env"):
    print("> .env failed to load, bad things might happen ...")


sess = sagemaker.Session()
role = "SageMakerTrainingRole"
# s3_train_data = "s3://sagemaker-us-east-1-941377158354/IndicPodcastsEn/"
s3_train_data = "s3://sagemaker-us-east-1-941377158354/nptel_109106147/train/"
output_path = f"s3://{sess.default_bucket()}/chatterbox-finetune-t3-models/"
checkpoint_s3_uri = f"s3://{sess.default_bucket()}/chatterbox-finetune-t3-checkpoints/"

hyperparameters = {
    "output_dir": "/opt/ml/model",
    "model_name_or_path": "ResembleAI/chatterbox",
    "load_from_disk_dir": "/opt/ml/input/data/train",
    "eval_split_size": 0.001,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 5e-5,
    "warmup_steps": 200,
    "logging_steps": 16,
    "eval_strategy": "steps",
    "eval_steps": 1000,
    "save_strategy": "steps",
    "save_steps": 1000,
    "save_total_limit": 4,
    "fp16": True,
    "report_to": "wandb",
    "dataloader_num_workers": 8,
    "do_train": True,
    "do_eval": True,
    "dataloader_pin_memory": False,
    "eval_on_start": True,
    "label_names": "labels_speech",
    "text_column_name": "text",
}

max_run_seconds = 24 * 3600
max_wait_seconds = 30 * 3600


estimator = PyTorch(
    entry_point="src/finetune_t3.py",
    source_dir=".",
    role=role,
    framework_version="2.2.0",
    py_version="py310",
    # image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04",
    instance_count=1,
    instance_type="ml.p3dn.24xlarge",
    volume_size=1024,
    hyperparameters=hyperparameters,
    dependencies=["requirements.txt"],
    distribution={
        "smdistributed": {
            "dataparallel": {"enabled": True}
        }
    },
    output_path=output_path,
    base_job_name="chatterbox-finetune-t3",
    use_spot_instances=True,
    max_run=max_run_seconds,
    max_wait=max_wait_seconds,
    checkpoint_s3_uri=checkpoint_s3_uri,
    environment={
        "NCCL_DEBUG": "INFO",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "MIXED_PRECISION": os.environ.get("MIXED_PRECISION", "no")
    },
)

estimator.fit({"train": s3_train_data})
