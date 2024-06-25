from transformers.pipelines.pt_utils import KeyDataset

from datasets import load_dataset

from transformers import pipeline, AutoTokenizer

from social_media_nlp.models.transformers.inference import load_language_model
from social_media_nlp.models.evaluation import compute_classification_metrics
from social_media_nlp.models.utils import output2label

import tqdm

import json
import os

import argparse

import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained_model_name_or_path", help="Model to fine-tune")

    args = parser.parse_args()
    pretrained_model_name_or_path = str(args.pretrained_model_name_or_path)

    # Load model
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    model_args = {
        "num_labels": 3,
        "id2label": id2label,
        "label2id": label2id,
    }

    model = load_language_model(
        pretrained_model_name_or_path,
        model_class="sequence-classification",
        model_args=model_args,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    # Load dataset
    dataset = load_dataset("tweet_eval", "sentiment")

    pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    # Evaluation
    start_time = time.time()

    result = [
        out
        for out in tqdm.tqdm(pipe(KeyDataset(dataset["test"], "text"), batch_size=8))
    ]

    end_time = time.time()
    execution_time = end_time - start_time

    y_pred = [output2label(x["label"]) for x in result]

    metrics = compute_classification_metrics(y_pred, dataset["test"]["label"])

    save_dir = (
        f"./models/tweet_eval/predictions/tweet_eval/{pretrained_model_name_or_path}/"
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(
        save_dir + "predictions.json",
        "w",
    ) as f:
        json.dump(
            {
                "predictions": y_pred,
                "execution_time": execution_time,
                "metrics": metrics,
            },
            f,
        )


if __name__ == "__main__":
    main()
