#!/usr/bin/env python3

# Algorithmic Graph Theory ~ F 97 :: CA3 ~ Q1
# by Hadi Safari (hadi@hadisafari.ir)


from itertools import product, chain
from random import randint

import networkx as nx


def greedy_min_dominating_set(graph):
    """Finds a minimal dominating set for the graph `graph` using greedy approach."""
    dominating_set = set()
    neighbours = set()
    while len(set(graph) - dominating_set - neighbours):
        node = max(
            [(node, len(set(graph.neighbors(node)).difference(dominating_set.union(neighbours))))
                for node in set(graph.nodes()).difference(dominating_set)],
            key=lambda pair: pair[1]
        )[0]
        dominating_set.add(node)
        neighbours = neighbours.union(neighbours, set(graph.neighbors(node))).difference(dominating_set)
    return dominating_set


def greedy_directed_min_dominating_set(graph):
    """Finds a minimal dominating set for the digraph `graph` using greedy approach."""
    dominating_set = set()
    neighbours = set()
    while len(set(graph) - dominating_set - neighbours):
        node = max(
            [(node, len({edge[0] for edge in graph.edges() if edge[1] is node and edge[0] not in dominating_set.union(neighbours)}))
                for node in set(graph.nodes()).difference(dominating_set)],
            key=lambda pair: pair[1]
        )[0]
        dominating_set.add(node)
        neighbours = neighbours.union(neighbours, set(graph.neighbors(node))).difference(dominating_set)
    return dominating_set


def min_weighted_dominating_set(graph, weight=None):
    r"""Returns a dominating set that approximates the minimum weight node
    dominating set.

    from networkx (`nx.min_weighted_dominating_set`)

    ### Parameters
    * `G` : NetworkX graph
        Undirected graph.
    * `weight` : string
        The node attribute storing the weight of an edge. If provided,
        the node attribute with this key must be a number for each
        node. If not provided, each node is assumed to have weight one.
    ### Returns
    * `min_weight_dominating_set` : set
        A set of nodes, the sum of whose weights is no more than `(\log
        w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of
        each node in the graph and `w(V^*)` denotes the sum of the
        weights of each node in the minimum weight dominating set.
    ### Notes
    This algorithm computes an approximate minimum weighted dominating
    set for the graph `G`. The returned solution has weight `(\log
    w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of each
    node in the graph and `w(V^*)` denotes the sum of the weights of
    each node in the minimum weight dominating set for the graph.
    This implementation of the algorithm runs in $O(m)$ time, where $m$
    is the number of edges in the graph.
    ### References
    1. Vazirani, Vijay V. *Approximation Algorithms*. Springer Science & Business Media, 2001.
    """

    if len(graph) == 0:
        return set()

    # This is the dominating set that will eventually be returned.
    dom_set = set()

    def _cost(node_and_neighborhood):
        """Returns the cost-effectiveness of greedily choosing the given node.

        `node_and_neighborhood` is a two-tuple comprising a node and its closed neighborhood.
        """
        (v, neighborhood) = node_and_neighborhood
        return graph.nodes[v].get(weight, 1) / len(neighborhood - dom_set)

    # This is a set of all vertices not already covered by the dominating set.
    vertices = set(graph)
    # This is a dictionary mapping each node to the closed neighborhood of that node.
    neighborhoods = {v: {v} | set(graph[v]) for v in graph}

    # Continue until all vertices are adjacent to some node in the dominating set.
    while vertices:
        # Find the most cost-effective node to add, along with its closed neighborhood.
        dom_node, min_set = min(neighborhoods.items(), key=_cost)
        # Add the node to the dominating set and reduce the remaining set of nodes to cover.
        dom_set.add(dom_node)
        del neighborhoods[dom_node]
        vertices -= min_set

    return dom_set


def get_queen_graph(n):
    def _edges_from_node_list(nodes):
        return [tuple(set(edge)) for edge in product(nodes, nodes) if edge[0] < edge[1]]
    queen = nx.Graph()
    queen.add_nodes_from(range(n * n))
    queen.add_edges_from([(i * n + j, i * n + k) for i in range(n) for j in range(n) for k in range(j + 1, n)])  # horizontal edges
    queen.add_edges_from([(j * n + i, k * n + i) for i in range(n) for j in range(n) for k in range(j + 1, n)])  # vertical edges
    queen.add_edges_from(chain.from_iterable(_edges_from_node_list(
        [node for node in [(l + i) * n + i for i in range(n)] if n * n > node >= 0]
    ) for l in range(-n + 1, n)))  # major diagonal edges: j - i = l \in (-n, n)
    queen.add_edges_from(chain.from_iterable(_edges_from_node_list(
        [node for node in [(l - i) * n + i for i in range(n)] if n * n > node >= 0]
    ) for l in range(2 * n - 1)))  # minor diagonal edges: i + j = l \in [0, 2n-2]
    return queen


def main():
    # solution of part 1
    print(len(greedy_min_dominating_set(get_queen_graph(8))))
    print(len(min_weighted_dominating_set(get_queen_graph(8))))
    ########
    # solution of part 2
    print(len(greedy_directed_min_dominating_set(nx.gnp_random_graph(15, 0.5, directed=True))))
    ########
    ########
    # solution of part 4
    g = nx.gnp_random_graph(5, 0.5)
    nx.set_node_attributes(g, {node: randint(1, 10) for node in g.nodes}, 'weight')
    print(len(min_weighted_dominating_set(g, 'weight')))


if __name__ == '__main__':
    main()
