from typing import Dict, List, Union

from social_media_nlp.models.utils import label2output

# Sentiment Analysis Prompt Templates

SA_TEMPLATE = """
Classify the following text as either positive, neutral or negative sentiment.
Return label only without any other text.
{examples}
Text: {text}
Sentiment:"""

SA_TEMPLATE_COMPLETION = """
Classify the following text as either positive, neutral or negative sentiment.
Return label only without any other text.
{examples}
Text: {text}
### Sentiment:
"""

SA_EXAMPLE_TEMPLATE = """
Text: {text}
Sentiment: {sentiment}"""

SA_TEMPLATE_TRAINING_COMPLETION_ONLY = """
Classify the following text as either positive, neutral or negative sentiment.
Return label only without any other text.
{examples}
Text: {text}
### Sentiment: {sentiment}"""

# Base Prompt Templates

MODEL_PROMPT_TEMPLATE = {
    "microsoft/Phi-3-mini-4k-instruct": "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
    "mistralai/Mistral-7B-Instruct-v0.2": "<s>[INST] {prompt} [/INST]",
}

MODEL_FINETUNING_PROMPT_TEMPLATE = {
    "microsoft/Phi-3-mini-4k-instruct": "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n{input_text}<|end|>\n<|assistant|>\n{output}",
    "mistralai/Mistral-7B-Instruct-v0.2": "<s>[INST] {input_text} [/INST]\n {output} </s>",
}

# Auxiliary Functions


def get_model_prompt(pretrained_model_name_or_path: str, prompt: str) -> str:
    """
    Retrieves a model prompt based on the specified `pretrained_model_name_or_path`.

    Args:
        pretrained_model_name_or_path (str): Name or path of the pretrained model.
        prompt (str): The prompt to be formatted.

    Returns:
        str: The formatted prompt string.
    """
    return MODEL_PROMPT_TEMPLATE.get(pretrained_model_name_or_path, "{prompt}").format(
        prompt=prompt
    )


def get_finetuning_prompt(
    pretrained_model_name_or_path: str, input: str, output: str
) -> str:
    """
    Generates a fine-tuning prompt using the specified input and output texts.

    Args:
        pretrained_model_name_or_path (str): Name or path of the pretrained model.
        input (str): Input text for fine-tuning.
        output (str): Output text for fine-tuning.

    Returns:
        str: The formatted fine-tuning prompt string.
    """
    template = MODEL_FINETUNING_PROMPT_TEMPLATE[pretrained_model_name_or_path]
    return template.format(input_text=input, output=output)


def formatting_prompts_function(
    pretrained_model_name_or_path: str,
    example: Dict[str, Union[str, List[str]]],
    include_target: bool = True,
    completion_only: bool = False,
) -> str:
    """
    Formats prompts for training or completion based on the specified example and options.

    Args:
        pretrained_model_name_or_path (str): Name or path of the pretrained model.
        example (Dict[str, Union[str, List[str]]]): Example containing 'text' and 'label'.
        include_target (bool, optional): Include the target output. Defaults to True.
        completion_only (bool, optional): Use completion-only template. Defaults to False.

    Returns:
        str: The formatted prompt string.
    """
    # prompt_template = (
    #     SA_TEMPLATE_TRAINING
    #     if not completion_only
    #     else SA_TEMPLATE_TRAINING_COMPLETION_ONLY
    # )
    prompt_template = SA_TEMPLATE_TRAINING_COMPLETION_ONLY
    input = prompt_template.format(
        examples="",
        text=example["text"],
        sentiment="",
    )
    output = label2output(example["label"]) if include_target else ""
    text = get_finetuning_prompt(
        pretrained_model_name_or_path, input=input, output=output
    )
    return text


def formatting_func_completion_only(
    pretrained_model_name_or_path: str, example: Dict[str, List[str]]
) -> List[str]:
    """
    Formats completion-only prompts based on the specified example for multiple texts.

    Args:
        pretrained_model_name_or_path (str): Name or path of the pretrained model.
        example (Dict[str, List[str]]): Example containing lists of 'text' and 'label'.

    Returns:
        List[str]: List of formatted prompt strings for each text in the example.
    """
    output_texts = []

    for i in range(len(example["text"])):
        text = formatting_prompts_function(
            pretrained_model_name_or_path,
            {
                "text": example["text"][i],
                "label": example["label"][i],
            },
            completion_only=True,
        )
        output_texts.append(text)
    return output_texts
