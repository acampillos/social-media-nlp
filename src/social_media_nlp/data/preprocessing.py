from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from typing import List, Dict
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
from typing import Tuple
from social_media_nlp.data.cleaning import clean_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def tokenize(text: str) -> List[str]:
    """
    Tokenize the text into words.

    Args:
        text (str): Input text to be tokenized.

    Returns:
        List[str]: List of tokens.
    """
    return word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove stopwords from the list of tokens.

    Args:
        tokens (List[str]): List of tokens.

    Returns:
        List[str]: List of tokens without stopwords.
    """
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]


def stem(tokens: List[str]) -> List[str]:
    """
    Perform stemming on the list of tokens.

    Args:
        tokens (List[str]): List of tokens.

    Returns:
        List[str]: List of stemmed tokens.
    """
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens


def stratified_sampling(
    df: pd.DataFrame, label_column: str, sample_size: int, random_state: int = 42
) -> pd.DataFrame:
    """
    Creates a smaller dataset from the given pandas DataFrame in a
    stratified manner based on the labels.

    Parameters:
        df (pandas DataFrame): Input DataFrame.
        label_column (str): Name of the column containing the labels.
        sample_size (int): Size of the smaller dataset to be created.
        random_state (int or None, optional): Controls the randomness of the sampling.
            Pass an int for reproducible output across multiple function calls.
            Defaults to None.

    Returns:
        sampled_df (pandas DataFrame): Sampled DataFrame.
    """
    X = df.drop(columns=[label_column])
    y = df[label_column]
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=sample_size, stratify=y, random_state=random_state
    )
    sampled_df = pd.concat([X_train, y_train], axis=1)
    return sampled_df


def sample_few_shot_examples(
    dataset: Dataset, k: int, random_state: int = 42
) -> List[Dict]:
    """
    Balanced sampling of examples for few-shot prompting.
    Selects k examples per class from the dataset.

    Args:
        dataset (Dataset): The dataset containing examples.
        k (int): The total number of examples to select per class.
        random_state (int, optional): The random seed for reproducibility.
            Defaults to 42.

    Returns:
        List[Dict]: A list of dictionaries with the examples' data.
    """
    df = pd.DataFrame(dataset)
    grouped = df.groupby("label")

    num_labels = len(grouped)
    examples_per_label = k // num_labels

    selected_examples = grouped.apply(
        lambda x: x.sample(examples_per_label, random_state=random_state)
    ).reset_index(drop=True)

    remaining = k - (examples_per_label * num_labels)
    if remaining > 0:
        remaining_examples = df.drop(selected_examples.index)
        selected_examples = pd.concat(
            [selected_examples, remaining_examples.sample(remaining)]
        )

    return selected_examples.to_dict("records")


def preprocess(text: str) -> List[str]:
    """
    Preprocess the text by applying cleaning, tokenization, stopwords removal
    and stemming.

    Args:
        text (str): Input text to be preprocessed.

    Returns:
        List[str]: List of preprocessed tokens.
    """
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    tokens = remove_stopwords(tokens)
    preprocessed_tokens = stem(tokens)
    return " ".join(preprocessed_tokens)


def extract_features(texts: List[str], method: str = "tfidf") -> Tuple:
    """
    Extract features from text using either count vectorization, TF-IDF or embeddings.

    Args:
        texts (List[str]): List of preprocessed texts.
        method (str): Feature extraction method (count, tfidf, embedding)

    Returns:
        Tuple: Feature matrix and feature extractor.
    """
    if method == "count":
        vectorizer = CountVectorizer()
    elif method == "tfidf":
        vectorizer = TfidfVectorizer()
    elif method == "embedding":
        model = SentenceTransformer("all-mpnet-base-v2")
        features = model.encode(texts, show_progress_bar=True)
        return features, model
    else:
        raise ValueError("Invalid method.")

    features = vectorizer.fit_transform(texts)
    return features, vectorizer
