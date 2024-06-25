from typing import List, Tuple, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MeanShift, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from wordsegment import load, segment
from sklearn.preprocessing import StandardScaler


class HashtagClusterer:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        load()

    def _avg_embedding_hashtags(self, hashtags: List[str]) -> np.ndarray:
        """Calculate the average embedding of hashtags.

        Args:
            hashtags (List[str]): List of hashtags.

        Returns:
            np.ndarray: Average embedding of the input hashtags.
        """
        if isinstance(hashtags, float) and np.isnan(hashtags):
            return np.array([])
        splitted_hashtags = [" ".join(segment(str(hashtag))) for hashtag in hashtags]
        embeddings = np.array(self.model.encode(splitted_hashtags))
        return np.sum(embeddings, axis=0) / embeddings.shape[0]

    def _segment_hashtags(self, hashtags: List[str]) -> List[str]:
        """Segment hashtags into individual words and returns them joined.

        Args:
            hashtags (List[str]): List of hashtags.

        Returns:
            List[str]: List of segmented hashtags.
        """
        return [" ".join(segment(hashtag)) for hashtag in hashtags]

    def cluster_hashtags(
        self, hashtags: List[str], model_params: Dict = None
    ) -> Tuple[MeanShift | KMeans, List[int], np.ndarray]:
        """
        Clusters hashtags based on their embeddings using either MeanShift or KMeans clustering.

        Args:
            hashtags (List[str]): List of hashtags to cluster.
            model_params (Dict[str, Union[int, float]], optional): Parameters for the model.
                If None, defaults to an empty dictionary. Should include 'n_clusters' for KMeans.
                Defaults to None.

        Returns:
            Tuple[Union[MeanShift, KMeans], List[int], np.ndarray]: A tuple containing:
                - The clustering model used (either MeanShift or KMeans).
                - Cluster labels assigned to each hashtag.
                - Embeddings of the hashtags after scaling.
        """
        if model_params is None:
            model_params = {}

        segmented_hashtags = self._segment_hashtags(hashtags)
        embeddings = self.model.encode(segmented_hashtags, convert_to_numpy=True)

        if "n_clusters" in model_params:
            clustering_model = KMeans(**model_params)
        else:
            clustering_model = MeanShift(**model_params)

        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

        cluster_labels = clustering_model.fit_predict(embeddings)
        return clustering_model, cluster_labels, embeddings

    def search_hashtags_within_cluster(
        self,
        hashtags: List[str],
        query_hashtag: str,
        clustering_model: MeanShift | KMeans,
        cluster_labels: List[int],
        hashtag_embeddings: np.ndarray,
        k: int = -1,
        threshold: float = 0.5,
    ) -> List[str]:
        """
        Searches for hashtags within the cluster of a given query hashtag based on
        cosine similarity of embeddings.

        Args:
            hashtags (List[str]): List of hashtags corresponding to each embedding.
            query_hashtag (str): The hashtag to use as the query.
            clustering_model (MeanShift or KMeans): Clustering model used to group hashtags.
            cluster_labels (List[int]): Cluster labels corresponding to each hashtag.
            hashtag_embeddings (np.ndarray): Embeddings of hashtags,
                where each row corresponds to a hashtag.
            k (int, optional): Maximum number of hashtags to return. Defaults to -1 (return all).
            threshold (float, optional): Minimum similarity to consider a hashtag relevant.
                Defaults to 0.5.

        Returns:
            List[str]: List of most relevant hashtags based on similarity to the query hashtag.
        """
        segmented_query_hashtag = " ".join(segment(query_hashtag))
        query_embedding = np.array(self.model.encode(segmented_query_hashtag))

        most_relevant_cluster = clustering_model.predict(
            query_embedding.reshape(1, -1)
        )[0]

        cluster_indices = [
            i
            for i, label in enumerate(cluster_labels)
            if label == most_relevant_cluster and hashtags[i] != query_hashtag
        ]
        similarities = {}

        for idx in cluster_indices:
            hashtag = hashtags[idx]
            embedding = hashtag_embeddings[idx]
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities[hashtag] = similarity

        sorted_hashtags = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        filtered_hashtags = [
            (tag, similarity)
            for tag, similarity in sorted_hashtags
            if similarity >= threshold
        ]
        return filtered_hashtags[:k]
