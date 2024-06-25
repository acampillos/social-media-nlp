import re


def clean_text(text: str) -> str:
    """
    Clean the text by removing special characters, numbers, and converting to lowercase.

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text


def remove_urls(text: str) -> str:
    """
    Remove URLs from a given text string.

    Args:
        text (str): The text from which URLs need to be removed.

    Returns:
        str: The text with URLs removed.
    """
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)
