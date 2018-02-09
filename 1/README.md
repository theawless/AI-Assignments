# Assignment 1: Feature Selection using searching techniques

### Submission by Group **4 horsemen**:
* Abhinav Singh 140101002
* Chetna Warkade 140101016
* Subham Gupta 140101072
* Rahul Kant 140101053

We have implemented the following search strategies.

1. Hill Climbing Technique
  ```python
def hill_climbing(M, J, get_neighbours=None):
    """
    Start at a random position and greedily search the neighbourhood.
    :param M: number of features
    :param J: cost function
    :param get_neighbours: neighbour operation
    :return: indices, score
    """
```

2. Sequential Forward Selection (SFS)
  ```python
def sequential_forward(M, J, target_cost):
    """
    Start from an empty set and greedily add features.
    :param M: number of features
    :param J: cost function
    :param target_cost: stop the search when reached
    :return: indices, score
    """
```

The code is well documented and can be run using `python3 main.py`.
main.py shows the comparison between hill_climbing, sequentional_forward, and exhaustive search methods on diabetes data set and friedman dataset (both available in scikit-learn). We use linear regression and lasso regression as evaluation metrics.
Along with the indices for features to be chosen, all techniques return the scores calculated using the cost function.

For using new data sets, one needs to provide the arguments as described in the documentation above (and in code selection.py). The search techniques take J (the cost function) as a closure, so as to hide the dataset details from feature selection techniques.
