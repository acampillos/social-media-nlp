from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


class SimilarityCalculator:

    @staticmethod
    def compute_similarity(df: pd.DataFrame, query: str, similarity_columns: List[str]):
        """
        Computes similarity scores between elements in a DataFrame and a given query.

        Args:
            df (pd.DataFrame): DataFrame containing the elements.
            query (str): The query string.
            similarity_columns (List[str]): List of column names containing elements to compute similarity on.

        Returns:
            pd.DataFrame: DataFrame with computed similarity scores.
        """
        output_columns = [f"{col}_cos_sim" for col in similarity_columns]
        for col, output_col in zip(similarity_columns, output_columns):
            df = SimilarityCalculator.compute_mean_similarity(
                df, query, col, output_col
            )
        df["cos_sim"] = df[output_columns].mean(axis=1)
        return df

    @staticmethod
    def compute_mean_similarity(
        profile_df: pd.DataFrame, query: str, column: str, output_column: str
    ):
        """
        Computes the mean similarity scores between elements and a given query.

        Args:
            profile_df (pd.DataFrame): DataFrame containing the elements.
            query (str): The query string.
            column (str): Column name containing elements to compute similarity on.
            output_column (str): Column name to store the computed similarity scores.

        Returns:
            pd.DataFrame: DataFrame with added column of computed similarity scores.
        """
        query_embeddings = get_embeddings([query])

        elements_embeddings = get_embeddings(
            profile_df[column].explode().dropna().tolist()
        )

        if not isinstance(profile_df[column].tolist()[0], list):
            profile_df[column] = profile_df[column].apply(lambda x: [x])

        profile_df[output_column] = profile_df[column].apply(
            lambda elements: cos_sim(
                [query], elements, query_embeddings, elements_embeddings
            )
        )
        profile_df[output_column] = profile_df[output_column].apply(np.mean)
        return profile_df


def get_embeddings(
    elements: List[str],
    model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2"),
) -> Dict[str, np.ndarray]:
    """Computes the embeddings of the given elements using a SentenceTransformer's model.

    Args:
        elements (List[str]): Elements to compute the embeddings.
        model (SentenceTransformer, optional): SentenceTransfomer's model to compute embeddings.
            Defaults to SentenceTransformer("all-MiniLM-L6-v2").

    Returns:
        Dict[str, np.ndarray]: Dictionary with element-embedding pairs.
    """
    unique_elements = np.unique(elements)

    embeddings = model.encode(unique_elements)
    embeddings_dict = {
        unique_elements[i]: embeddings[i] for i in range(len(unique_elements))
    }

    return embeddings_dict


def cos_sim(
    a: List[str],
    b: List[str],
    embeddings_a: Dict[str, np.ndarray],
    embeddings_b: Dict[str, np.ndarray],
) -> List[float]:
    """Computes cosine similarity between two lists using their embeddings.
    Embeddings must have the list elements as keys and embeddings as values.

    Args:
        a (List[str]): First list of elements to compute cosine similarity.
        b (List[str]): Second list of elements to compute cosine similarity.
        embeddings_a (Dict[str, np.ndarray]): Embeddings of list 'a'.
        embeddings_b (Dict[str, np.ndarray]): Embeddings of list 'b'.

    Returns:
        List[float]: Cosine similarity values for a and b elements pairs.
    """
    return [float(util.cos_sim(embeddings_a[x], embeddings_b[y])) for x in a for y in b]
