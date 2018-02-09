"""
Implements various feature subset selection techniques.
"""

import math
import random
from itertools import chain, combinations

random.seed(0)


def exhaustive(M, J):
    """
    Finds the set of indices that maximises cost from the powerset.
    :param M: number of features
    :param J: cost function
    :return: indices
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
            best_cost = cost
            best_indices = indices
    return best_indices


def sequential_forward(M, J, target_cost):
    """
    Start from an empty set and greedily add features.
    :param M: number of features
    :param J: cost function
    :param target_cost: stop the search when reached
    :return: indices
    """

    indices, best_indices = list(range(M)), []
    while indices:
        random.shuffle(indices)
        best_cost, best_index = -math.inf, None
        for index in indices:
            best_indices.append(index)
            cost = J(best_indices)
            best_indices.pop()
            if best_cost < cost:
                best_cost = cost
                best_index = index
        indices.pop(best_index)
        best_indices.append(best_index)
        if best_cost >= target_cost:
            return best_indices
    return best_indices
