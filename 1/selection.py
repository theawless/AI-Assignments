"""
Implements various feature subset selection techniques.
"""

import math
import random
from itertools import chain, combinations

random.seed(0)


def hill_climbing(M, J, get_neighbours=None):
    """
    Start at a random position and greedily search the neighbourhood.
    :param M: number of features
    :param J: cost function
    :param get_neighbours: neighbour operation
    :return: indices, score
    """

    def get_unit_neighbours(indices):
        """
        Gets neighbours at unit distance from current indices.
        :param indices: current node
        :return: next nodes
        """
        # remove left and right elements
        neighbours = [indices[1:], indices[:-1]]
        # add left element
        if indices[0] > 1:
            neighbour = [indices[0] - 1] + indices
            neighbours.append(neighbour)
        # add right element
        if indices[-1] < M - 1:
            neighbour = indices + [indices[-1] + 1]
            neighbours.append(neighbour)
        return neighbours

    if not get_neighbours:
        get_neighbours = get_unit_neighbours

    indices, next_indices = [], sorted(random.sample(list(range(M)), random.randint(0, M - 1)))
    cost, next_cost = -math.inf, J(next_indices)
    while cost < next_cost:
        cost, indices = next_cost, next_indices
        for next_indices_temp in get_neighbours(indices):
            next_cost_temp = J(next_indices_temp)
            if next_cost < next_cost_temp:
                next_cost, next_indices = next_cost_temp, next_indices_temp
    return indices, cost


def sequential_forward(M, J, target_cost):
    """
    Start from an empty set and greedily add features.
    :param M: number of features
    :param J: cost function
    :param target_cost: stop the search when reached
    :return: indices, score
    """

    indices, best_indices = list(range(M)), []
    while indices:
        best_cost, best_index = -math.inf, None
        for index in indices:
            best_indices.append(index)
            cost = J(best_indices)
            best_indices.pop()
            if best_cost < cost:
                best_cost, best_index = cost, index
        indices.pop(best_index)
        best_indices.append(best_index)
        if best_cost >= target_cost:
            return best_indices, best_cost
    return best_indices, J(best_indices)


def exhaustive(M, J):
    """
    Finds the set of indices that maximises cost from the powerset.
    :param M: number of features
    :param J: cost function
    :return: indices, score
    """

    def power_set(iterable):
        """
        Finds power set. Taken from https://docs.python.org/3/library/itertools.html
        :param iterable: set
        :return: powerset
        """

        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    best_cost, best_indices = -math.inf, []
    for indices in power_set(range(M)):
        cost = J(indices)
        if best_cost < cost:
            best_cost, best_indices = cost, indices
    return best_indices, best_cost
