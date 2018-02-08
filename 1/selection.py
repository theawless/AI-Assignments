import math
import random
from collections import deque
from itertools import chain, combinations

random.seed(0)


def exhaustive(M, J):
    # taken from https://docs.python.org/3/library/itertools.html
    def power_set(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    best_cost, best_indices = -math.inf, []
    for indices in power_set(range(M)):
        if not indices:
            continue
        cost = J(indices)
        if best_cost < cost:
            best_cost = cost
            best_indices = indices
    return best_indices


def sequential_forward(M, J, target_cost):
    indices = deque(range(M))
    random.shuffle(indices)

    best_cost, best_indices = -math.inf, []
    while indices:
        index = indices.popleft()
        best_indices.append(index)
        cost = J(best_indices)
        if best_cost < cost:
            best_cost = cost
        else:
            indices.append(index)
            best_indices.pop()
        if best_cost >= target_cost:
            return best_indices
