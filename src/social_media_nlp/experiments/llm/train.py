import torch._dynamo

import mlflow

import pandas as pd

from datasets import load_dataset, DatasetDict, Dataset

from social_media_nlp.models.transformers.train import train
from social_media_nlp.data.preprocessing import stratified_sampling
from social_media_nlp.prompts import formatting_func_completion_only

from pathlib import Path
import os

import logging
import argparse

import time


def main():
    torch._dynamo.config.suppress_errors = True

    logging.getLogger("transformers").setLevel(logging.ERROR)

    mlflow.autolog()

    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="Model to fine-tune")
    parser.add_argument("-r", "--lora_rank", help="LoRA rank")
    parser.add_argument("-a", "--lora_alpha", help="LoRA alpha")
    parser.add_argument("-e", "--epochs", help="Number of epochs")
    parser.add_argument("-t", "--train_samples", help="Number of training samples")
    parser.add_argument("-v", "--val_samples", help="Number of validation samples")

    args = parser.parse_args()

    pretrained_model_name_or_path = str(args.model_id)
    r = int(args.lora_rank)
    alpha = int(args.lora_alpha)
    epochs = int(args.epochs)
    samples_train = int(args.train_samples)
    samples_val = int(args.val_samples)

    output_dir = Path("./models/")
    log_dir = Path("./logs/")

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # MLflow setup
    mlflow.set_experiment(
        f"tweet_eval/{pretrained_model_name_or_path}/{time.strftime('%Y-%m-%d')}"
    )
    mlflow.enable_system_metrics_logging()

    # Training setup
    batch_size = 1

    lora_config = {
        "r": r,
        "lora_alpha": alpha,
        "bias": "none",
        "lora_dropout": 0.05,
    }

    peft_training_args = {
        "output_dir": output_dir / f"tweet_eval/{pretrained_model_name_or_path}",
        "overwrite_output_dir": True,
        "logging_dir": log_dir / f"tweet_eval/{pretrained_model_name_or_path}",
        "logging_steps": 10,
        "logging_strategy": "steps",
        "report_to": "mlflow",
        "save_strategy": "steps",
        "save_steps": 300,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "do_eval": True,
        "evaluation_strategy": "steps",
        "eval_accumulation_steps": 30,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "num_train_epochs": epochs,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "eval_steps": 300,
        "optim": "paged_adamw_8bit",
        "group_by_length": True,
        "torch_compile": True,
        "disable_tqdm": False,
        "fp16": True,
        "load_best_model_at_end": True,
    }

    collator = "DataCollatorForCompletionOnlyLM"

    peft_training_args["output_dir"] = (
        peft_training_args["output_dir"]
        / f"_rank_{lora_config['r']}_alpha_{lora_config['lora_alpha']}_epochs_{peft_training_args['num_train_epochs']}_train_{samples_train}_val_{samples_val}"
    )
    peft_training_args["logging_dir"] = (
        peft_training_args["logging_dir"]
        / f"_rank_{lora_config['r']}_alpha_{lora_config['lora_alpha']}_epochs_{peft_training_args['num_train_epochs']}_train_{samples_train}_val_{samples_val}"
    )
    peft_training_args["run_name"] = (
        f"rank_{lora_config['r']}_alpha_{lora_config['lora_alpha']}_epochs_{peft_training_args['num_train_epochs']}_train_{samples_train}_val_{samples_val}"
    )

    peft_training_args["output_dir"].mkdir(parents=True, exist_ok=True)
    peft_training_args["logging_dir"].mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset("tweet_eval", "sentiment")

    dataset = dataset.map(
        lambda entry: {
            "prompt": entry,
        }
    )

    stratified_train_df = stratified_sampling(
        df=pd.DataFrame(dataset["train"]),
        label_column="label",
        sample_size=samples_train,
    )

    stratified_val_df = stratified_sampling(
        df=pd.DataFrame(dataset["validation"]),
        label_column="label",
        sample_size=samples_val,
    )

    processed_dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(stratified_train_df, preserve_index=False),
            "validation": Dataset.from_pandas(stratified_val_df, preserve_index=False),
        }
    )

    train(
        pretrained_model_name_or_path,
        lora_config,
        peft_training_args,
        collator,
        processed_dataset["train"],
        processed_dataset["validation"],
        trainer_args=(
            {}
            if collator == "DataCollatorForLanguageModeling"
            else {
                "formatting_func": lambda x: formatting_func_completion_only(
                    pretrained_model_name_or_path, x
                )
            }
        ),
    )


if __name__ == "__main__":
    main()
