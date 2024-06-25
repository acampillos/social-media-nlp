import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from datasets import load_dataset

from social_media_nlp.data.preprocessing import preprocess, extract_features
from social_media_nlp.models.evaluation import compute_classification_metrics
from social_media_nlp.models.hyperparameter_tuning import optimize_model, objective

import time

import os
import json


def main():
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    # Hyperparameters search space
    models = [
        {
            "model_class": LogisticRegression,
            "hyperparams": {
                "C": {"type": "float", "values": (0.1, 100.0)},
                "solver": {"type": "categorical", "values": ["liblinear", "saga"]},
                "penalty": {"type": "categorical", "values": ["l1", "l2"]},
            },
        },
        {
            "model_class": RandomForestClassifier,
            "hyperparams": {
                "n_estimators": {"type": "int", "values": (50, 500)},
                "max_depth": {"type": "int", "values": (10, 100)},
                "min_samples_split": {"type": "float", "values": (0.01, 0.5)},
                "max_features": {"type": "float", "values": (0.1, 1.0)},
            },
        },
    ]

    # Load dataset
    dataset = load_dataset("tweet_eval", "sentiment")

    df_train = dataset["train"].to_pandas()
    df_val = dataset["validation"].to_pandas()
    df_test = dataset["test"].to_pandas()

    # Train and evaluate models
    for vectorization_method in ["tfidf", "embedding"]:

        if vectorization_method in ["tfidf", "count"]:
            df_train["preprocessed_text"] = df_train["text"].apply(preprocess)
            df_val["preprocessed_text"] = df_val["text"].apply(preprocess)
            df_test["preprocessed_text"] = df_test["text"].apply(preprocess)
        else:
            df_train["preprocessed_text"] = df_train["text"]
            df_val["preprocessed_text"] = df_val["text"]
            df_test["preprocessed_text"] = df_test["text"]

        X_train_val, vectorizer = extract_features(
            pd.concat(
                [df_train["preprocessed_text"], df_val["preprocessed_text"]]
            ).tolist(),
            method=vectorization_method,
        )
        y_train_val = pd.concat([df_train["label"], df_val["label"]])

        if vectorization_method == "embedding":
            X_test = vectorizer.encode(df_test["preprocessed_text"].tolist())
        else:
            X_test = vectorizer.transform(df_test["preprocessed_text"])
        y_test = df_test["label"]

        for model_config in models:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = (
                f'tweet_eval/{model_config["model_class"].__name__}_{vectorization_method}'
            )
            best_params, _ = optimize_model(
                model_config,
                objective,
                X_train_val,
                y_train_val,
                study_name=f'{model_config["model_class"].__name__}_{vectorization_method}',
                n_trials=25 if vectorization_method == "embedding" else 50,
            )

            model_instance = model_config["model_class"](**best_params)

            start_time = time.time()
            model_instance.fit(X_train_val, y_train_val)
            end_time = time.time()
            execution_time = end_time - start_time

            y_pred = model_instance.predict(X_test).tolist()

            save_dir = f"./models/tweet_eval/predictions/tweet_eval/ml/{model_config['model_class'].__name__}/{vectorization_method}/"

            os.makedirs(save_dir, exist_ok=True)

            with open(
                save_dir + "predictions.json",
                "w",
            ) as f:
                json.dump(
                    {
                        "predictions": y_pred,
                        "execution_time": execution_time,
                        "metrics": compute_classification_metrics(
                            y_test.tolist(), y_pred
                        ),
                    },
                    f,
                )


if __name__ == "__main__":
    main()
