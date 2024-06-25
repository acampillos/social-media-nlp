def output2label(text: str) -> int:
    """
    Converts textual sentiment labels to numerical labels.

    Args:
        text (str): The text containing sentiment information.

    Returns:
        int: Numerical label corresponding to the sentiment.
             2 for "positive", 1 for "neutral", 0 for "negative" and 3 for non-existent labels.
    """
    if "positive" in text.lower():
        return 2
    elif "negative" in text.lower():
        return 0
    elif "neutral" in text.lower():
        return 1
    else:
        return 3  # non-existent label


def label2output(label: int) -> str:
    """
    Converts numerical sentiment labels to textual labels.

    Args:
        label (int): The numerical label representing the sentiment:
                     0 for "Negative", 1 for "Neutral", and 2 for "Positive".

    Returns:
        str: Textual label corresponding to the numerical sentiment label.
    """
    if label == 0:
        return "Negative"
    elif label == 1:
        return "Neutral"
    elif label == 2:
        return "Positive"
