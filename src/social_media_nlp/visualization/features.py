import re
from typing import List


def get_hashtags(text: str) -> List[str]:
    """Gets the hashtags inside a text (e.g.: #hashtag).

    Args:
        text (str): Text to find hashtags.

    Returns:
        List[str]: Hashtags found.
    """
    return re.findall(r"\#\w+", text)


def get_mentions(text: str) -> List[str]:
    """Gets the mentions in a text (e.g.: @username).

    Args:
        text (str): Text to find mentions.

    Returns:
        List[str]: Mentions found.
    """
    return re.findall(r"\@\w+", text)


def get_urls(text: str) -> List[str]:
    """Gets the urls of a text.

    Args:
        text (str): Text to get the urls from.

    Returns:
        List[str]: Urls found in text.
    """
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return re.findall(url_pattern, text)
