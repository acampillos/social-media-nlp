from torch.nn import Module
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig,
)
import gc
import torch
import bitsandbytes as bnb
from typing import Dict, Tuple
from numba import cuda

from peft import AutoPeftModelForCausalLM


def get_number_of_trainable_model_parameters(model: Module, bits: int = 4) -> str:
    """
    Calculate the number of trainable parameters in a model and the percentage
    they represent of the total parameters.

    Args:
        model (Module): The PyTorch model to analyze.
        bits (int, optional): Bit precision for trainable parameters. Default is 4.

    Returns:
        str: A formatted string with the number of trainable parameters,
             total parameters, and the percentage of trainable parameters.
    """
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()

    if bits == 4:
        trainable_model_params /= 2

    response = f"""
    Trainable model parameters: {trainable_model_params}
    All model parameters: {all_model_params}
    Percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"""
    return response


def get_special_tokens_model_tokenizer(model: Module, tokenizer: AutoTokenizer) -> str:
    """
    Retrieve special tokens information for both model and tokenizer.

    Args:
        model (Module): The PyTorch model to analyze.
        tokenizer (AutoTokenizer): The tokenizer used with the model.

    Returns:
        str: A formatted string with the special tokens and their IDs for both
             the tokenizer and the model.
    """
    response = f"""Special tokens:
    Tokenizer EOS: {tokenizer.eos_token}, {tokenizer.eos_token_id}
    Model EOS: {tokenizer.convert_ids_to_tokens(model.config.eos_token_id)}, {model.config.eos_token_id}

    Tokenizer BOS: {tokenizer.bos_token}, {tokenizer.bos_token_id}
    Model BOS: {tokenizer.convert_ids_to_tokens(model.config.bos_token_id)}, {model.config.bos_token_id}

    Tokenizer UNK: {tokenizer.unk_token}, {tokenizer.unk_token_id}
    Tokenizer PAD: {tokenizer.pad_token}, {tokenizer.pad_token_id}
    Model PAD: {tokenizer.convert_ids_to_tokens(model.config.pad_token_id)}, {model.config.pad_token_id}"""

    return response


def bytes_to_giga_bytes(bytes: int) -> float:
    """
    Convert bytes to gigabytes.

    Args:
        bytes (int): Size in bytes.

    Returns:
        float: Size in gigabytes.
    """
    return bytes / 1024 / 1024 / 1024


def clear_torch_cache():
    """
    Clear the PyTorch cache to free up GPU memory.
    """
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def release_gpu_memory():
    """
    Release GPU memory by resetting the current CUDA device.
    """
    device = cuda.select_device(0)
    device.reset()


def find_all_linear_names(bits: int, model: Module):
    """Find names of all modules in `model` that match a specific linear layer type
    based on the number of `bits`.

    Args:
        bits (int): Number of bits (4 or 8) specifying the linear layer type.
        model (torch.nn.Module): The model to search for linear layers.

    Returns:
        List[str]: List of unique module names that match the specified linear layer type.
    """
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_model_param_dtypes(model: Module):
    """Print parameter data types and their counts in the given `model`.

    Args:
        model (torch.nn.Module): The model to analyze.
    """
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    print("Param dtypes:")
    for k, v in dtypes.items():
        print("dtype: ", k, "n: ", v, "%: ", v / total)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and model embeddings to accommodate additional special tokens.

    Note: This function is not optimized and may not ensure embedding sizes divisible by 64.

    Args:
        special_tokens_dict (Dict[str, Union[str, int]]): Dictionary of special tokens to add.
        tokenizer (PreTrainedTokenizer): Tokenizer associated with the model.
        model (PreTrainedModel): Pretrained model to resize embeddings.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def merge_adapters(
    pretrained_model_name_or_path: str,
    output_path: str,
    max_shard_size: str = "2GB",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Merge adapters into a single model and save it along with the tokenizer.

    Args:
        pretrained_model_name_or_path (str): Pretrained model name or path.
        output_path (str): Output directory to save the merged model and tokenizer.
        max_shard_size (str, optional): Maximum shard size for model serialization.
            Defaults to "2GB".

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Tuple with the merged model and tokenizer.
    """
    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=compute_dtype,
        return_dict=False,
        low_cpu_mem_usage=True,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        output_path, safe_serialization=True, max_shard_size=max_shard_size
    )
    tokenizer.save_pretrained(output_path)

    return merged_model, tokenizer


def get_max_length(model: PreTrainedModel) -> int:
    """Retrieve the maximum sequence length from the model configuration.

    Args:
        model (PreTrainedModel): Pretrained model containing configuration attributes.

    Returns:
        int: Maximum length value found in the model configuration.
    """
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length
