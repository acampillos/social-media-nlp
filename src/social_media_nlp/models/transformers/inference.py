import torch

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from peft import AutoPeftModelForCausalLM


def load_language_model(
    pretrained_model_name_or_path: str,
    trust_remote_code=True,
    device_map="auto",
    model_args: dict = None,
    model_class: str = "causal-lm",
    merged_model: bool = False,
) -> torch.nn.Module:
    """
    Load a pre-trained language model with specified configurations.

    This function loads a language model using the specified parameters. It supports
    different model classes such as causal language modeling and sequence classification.

    Args:
        pretrained_model_name_or_path (str): Path to the pre-trained model or model identifier.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True.
        device_map (str, optional): Specifies the model allocation on devices. Defaults to "auto".
        model_args (dict, optional): Additional arguments to pass to the model. Defaults to None.
        model_class (str, optional): Type of model to load (causal-lm, sequence-classification).
        merged_model (bool, optional): Loads a merged model (only causal-lm). Defaults to False.

    Returns:
        torch.nn.Module: The loaded language model.
    """
    if model_args is None:
        model_args = {}

    if model_class == "causal-lm":
        if merged_model:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                **model_args,
            )
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                return_dict=False,
                **model_args,
            )
    elif model_class == "sequence-classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            **model_args,
        )
    return model
