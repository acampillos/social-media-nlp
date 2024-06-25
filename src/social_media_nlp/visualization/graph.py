import networkx as nx
from networkx import Graph
from typing import List, Tuple, Any


def shortest_path(graph: Graph, source: str, target: str) -> List[str]:
    """
    Computes the shortest path between two nodes in the graph.

    Parameters:
        graph (Graph): The graph to compute the shortest path on.
        source (str): The starting node.
        target (str): The ending node.

    Returns:
        List[str]: A list of nodes representing the shortest path from source to target.
    """
    return nx.shortest_path(graph, source=source, target=target)


def top_neighbors(graph: Graph, query: str, n: int = 5) -> List[Tuple[str, int]]:
    """
    Finds the top neighbors of a given node based on their degrees.

    Parameters:
        graph (Graph): The graph to find neighbors in.
        query (str): The node to find neighbors for.
        n (int): The number of top neighbors to return. Defaults to 5.

    Returns:
        List[Tuple[str, int]]: A list of tuples containing the neighbor node and its degree.
    """
    neighbors = list(graph.neighbors(query))
    neighbor_degrees = [(neighbor, graph.degree(neighbor)) for neighbor in neighbors]
    sorted_neighbors = sorted(neighbor_degrees, key=lambda x: x[1], reverse=True)[:n]
    return sorted_neighbors


def top_similar(graph: Graph, query: str, n: int = 5) -> List[Tuple[str, str, float]]:
    """
    Finds the top similar nodes to a given node based on Jaccard similarity.

    Parameters:
        graph (Graph): The graph to find similar nodes in.
        query (str): The node to find similar nodes for.
        n (int): The number of top similar nodes to return. Defaults to 5.

    Returns:
        List[Tuple[str, str, float]]: A list of tuples containing the similar node,
            similarity score and target node.
    """
    jaccard_similarities = nx.jaccard_coefficient(
        graph, [(query, neighbor) for neighbor in graph.neighbors(query)]
    )
    sorted_similarities = sorted(
        jaccard_similarities, key=lambda x: x[2], reverse=True
    )[:n]
    return sorted_similarities


def community_detection(graph: Graph) -> List[Any]:
    """
    Detects communities in the graph using greedy modularity optimization.

    Parameters:
        graph (Graph): The graph to detect communities in.

    Returns:
        List[Any]: A list of communities identified in the graph.
    """
    communities = nx.algorithms.community.greedy_modularity_communities(graph)
    return communities


def collaborative_filtering_recommendation(
    graph: Graph, user_interactions: List[str], n: int = 5
) -> List[Tuple[str, int]]:
    """
    Provides collaborative filtering recommendations based on user interactions.

    Parameters:
        graph (Graph): The graph to base recommendations on.
        user_interactions (List[str]): A list of nodes representing user interactions.
        n (int): The number of recommendations to provide. Defaults to 5.

    Returns:
        List[Tuple[str, int]]: A list of recommended nodes and their scores.
    """
    similar_queries = []
    for interaction in user_interactions:
        similar_queries.extend([sim[1] for sim in top_similar(graph, interaction, n)])

    similar_queries = list(set(similar_queries) - set(user_interactions))
    hashtag_scores = {query: 0 for query in similar_queries}
    for interaction in user_interactions:
        for query in similar_queries:
            if query in graph.neighbors(interaction):
                hashtag_scores[query] += 1

    sorted_queries = sorted(hashtag_scores.items(), key=lambda x: x[1], reverse=True)[
        :n
    ]
    return sorted_queries


def influence_analysis(graph: Graph, query: str) -> float:
    """
    Calculates the influence of a node in the graph based on degree centrality.

    Parameters:
        graph (Graph): The graph to analyze.
        query (str): The node to analyze.

    Returns:
        float: The influence score of the node.
    """
    degree_centrality = nx.degree_centrality(graph)
    return degree_centrality.get(query, 0)
