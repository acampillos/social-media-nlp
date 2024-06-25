import torch

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from typing import Dict, Any

from datasets import Dataset
import evaluate

import numpy as np
import mlflow

from social_media_nlp.models.transformers.inference import load_language_model
from social_media_nlp.models.transformers.utils import (
    find_all_linear_names,
    smart_tokenizer_and_embedding_resize,
)


class MLflowLoggingCallback(TrainerCallback):
    """A callback class to log metrics to MLflow during training with the Trainer class.

    This class extends the `TrainerCallback` and overrides the `on_log` method to log evaluation
    metrics to MLflow.

    Methods:
        on_log(args, state, control, logs=None, **kwargs): Logs evaluation metrics to MLflow.
    """

    def __init__(self):
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Logs evaluation metrics to MLflow.

        Args:
            args: Training arguments.
            state: The state of the trainer.
            control: The control object for the trainer.
            logs (dict, optional): A dictionary of logs containing metrics to be logged.
                Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if logs is not None:
            for key, value in logs.items():
                if key.startswith("eval_"):
                    mlflow.log_metric(key, value, step=state.global_step)


def train_seq_lm(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    training_args: Dict[str, Any],
    train_dataset: Dataset,
    validation_dataset: Dataset,
):
    """Trains a sequence language model using the Hugging Face Trainer API.

    Args:
        model (torch.nn.Module): The language model to be trained.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        training_args (Dict[str, Any]): A dictionary of training arguments.
        train_dataset (Dataset): The training dataset.
        validation_dataset (Dataset): The validation dataset.
    """

    def compute_metrics(eval_pred) -> dict:
        """Computes evaluation metrics.

        Args:
            eval_pred: A tuple containing predictions and labels.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metric = evaluate.load("f1")

    output_dir = training_args["output_dir"]
    training_args = TrainingArguments(**training_args)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[MLflowLoggingCallback()],
    )
    trainer.train()
    trainer.save_model(output_dir=f"{output_dir}/best/")


def train(
    pretrained_model_name_or_path: str,
    lora_config: Dict[str, Any],
    training_args: Dict[str, Any],
    collator: str,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    max_shard_size: str = "2GB",
    trainer_args: Dict = None,
):
    """Trains an LLM using the Hugging Face Trainer API.

    Args:
        pretrained_model_name_or_path (str): The path to the pre-trained model or its name.
        lora_config (Dict[str, Any]): A dictionary containing the LoRA configuration.
        training_args (Dict[str, Any]): A dictionary containing the training arguments.
        collator (str): The data collator type to be used
            (DataCollatorForCompletionOnlyLM, DataCollatorForLanguageModeling)
        train_dataset (Dataset): The training dataset.
        validation_dataset (Dataset): The validation dataset.
        max_shard_size (str, optional): Maximum shard size for saving the model. Defaults to "2GB".
        trainer_args (Dict, optional): Additional arguments for the trainer. Defaults to None.
    """
    if trainer_args is None:
        trainer_args = {}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = load_language_model(
        pretrained_model_name_or_path,
        model_args={
            "quantization_config": bnb_config,
            "low_cpu_mem_usage": True,
            "attn_implementation": "flash_attention_2",
        },
        merged_model=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    special_tokens_dict = {"pad_token": tokenizer.eos_token}
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    original_model = prepare_model_for_kbit_training(model)
    original_model.gradient_checkpointing_enable()

    config = LoraConfig(
        target_modules=find_all_linear_names(4, original_model),
        task_type="CAUSAL_LM",
        **lora_config,
    )

    peft_model = get_peft_model(original_model, config)

    peft_training_args = TrainingArguments(**training_args)

    if collator == "DataCollatorForCompletionOnlyLM":
        response_template_ids = tokenizer.encode(
            "### Sentiment:", add_special_tokens=False
        )[1:]
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
        )
    elif collator == "DataCollatorForLanguageModeling":
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        trainer_args["packing"] = True
        trainer_args["dataset_text_field"] = "prompt"

    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=peft_training_args,
        peft_config=config,
        data_collator=data_collator,
        max_seq_length=2048,
        **trainer_args,
    )
    trainer.train()

    # trainer.save_model(output_dir=training_args["output_dir"])
    trainer.model.save_pretrained(
        save_directory=training_args["output_dir"], max_shard_size=max_shard_size
    )
    tokenizer.save_pretrained(save_directory=training_args["output_dir"])

    del model
    del trainer
    torch.cuda.empty_cache()
