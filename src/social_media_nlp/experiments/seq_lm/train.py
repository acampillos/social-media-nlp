import mlflow

from typing import Any, Dict

from datasets import load_dataset

from transformers import AutoTokenizer

from social_media_nlp.data.cleaning import clean_text, remove_urls
from social_media_nlp.models.transformers.train import train_seq_lm
from social_media_nlp.models.transformers.inference import load_language_model

from pathlib import Path
import argparse


def main():

    def preprocess_data(entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses the text in the given entry dictionary.

        Args:
            entry (dict): A dictionary containing at least the key 'text' with a string value.

        Returns:
            dict: The input dictionary with the 'text' field modified after processing.
        """
        entry["text"] = clean_text(entry["text"])
        entry["text"] = remove_urls(entry["text"])
        entry["text"] = entry["text"].lower()
        return entry

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenizes the text in the given batch dictionary using a specified tokenizer.

        Args:
            batch (dict): A dictionary containing at least the key 'text' with a string value.

        Returns:
            dict: The dictionary with tokenized 'text' field added processed.
        """
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="Model to fine-tune")
    parser.add_argument("num_train_epochs", help="Number of epochs")

    args = parser.parse_args()
    pretrained_model_name_or_path = str(args.model_id)
    num_train_epochs = int(args.num_train_epochs)

    output_dir = Path("./models/")
    log_dir = Path("./logs/")

    # MLflow setup
    mlflow.autolog()
    mlflow.set_experiment(f"tweet_eval/{pretrained_model_name_or_path}")
    mlflow.enable_system_metrics_logging()

    # Model setup
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    model_args = {
        "num_labels": 3,
        "id2label": id2label,
        "label2id": label2id,
    }

    # Training setup
    batch_size = 8

    training_args = {
        "output_dir": output_dir / f"tweet_eval/{pretrained_model_name_or_path}",
        "logging_dir": log_dir / f"tweet_eval/{pretrained_model_name_or_path}",
        "do_eval": True,
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": 3e-5,
        "num_train_epochs": num_train_epochs,
        "warmup_ratio": 0.1,
        "logging_steps": 100,
        "save_strategy": "steps",
        "save_steps": 3000,
        "eval_steps": 1000,
        "optim": "adamw_torch_fused",
        "report_to": "mlflow",
        "run_name": pretrained_model_name_or_path,
        "torch_compile": True,
        "torch_compile_backend": "eager",
        "disable_tqdm": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "lr_scheduler_type": "cosine",
    }

    # Load model
    model = load_language_model(
        pretrained_model_name_or_path,
        model_class="sequence-classification",
        model_args=model_args,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    # Load dataset
    dataset = load_dataset("tweet_eval", "sentiment")

    tokenized_datasets = dataset.map(preprocess_data, batched=False).map(
        tokenize, batched=True
    )

    max_length = max(
        [
            len(tokenized_datasets["train"][i]["input_ids"])
            for i in range(len(tokenized_datasets["train"]))
        ]
    )
    model_args["max_length"] = min(model.config.max_position_embeddings, max_length)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    training_args["output_dir"] = (
        training_args["output_dir"] / f"_epochs_{num_train_epochs}"
    )
    training_args["logging_dir"] = (
        training_args["logging_dir"] / f"_epochs_{num_train_epochs}"
    )
    training_args["run_name"] = f"epochs_{num_train_epochs}"

    train_seq_lm(
        model,
        tokenizer,
        training_args=training_args,
        train_dataset=tokenized_datasets["train"],
        validation_dataset=tokenized_datasets["validation"],
    )


if __name__ == "__main__":
    main()
