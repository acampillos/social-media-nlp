def map_vader_score_to_label(score: float) -> int:
    """
    Maps a VADER sentiment score to a label based on predefined thresholds.
    Sentiment labels are defined as follows:
        - 2: Positive sentiment (score >= 0.05).
        - 1: Neutral sentiment (-0.05 < score < 0.05).
        - 0: Negative sentiment (score <= -0.05).

    Parameters:
        score (float): The VADER sentiment score to be mapped.

    Returns:
        int: The mapped sentiment label.
    """
    if score >= 0.05:
        return 2
    elif score <= -0.05:
        return 0
    else:
        return 1
