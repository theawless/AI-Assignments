import matplotlib.pyplot
import numpy


def errorize(observation, error):
    """
    Adds error to the observation.
    :param observation: original observation
    :param error: error probability
    :return: errored observation
    """
    errored = observation
    for i in range(len(errored)):
        if numpy.random.rand() < error:
            errored[i] = (errored[i] + 1) % 2
    return errored


def sensor_model(location, states, moves, error):
    """
    Gets the sensor model.
    :param location: current location
    :param states: possible states
    :param moves: possible moves
    :param error: error probability
    :return: sensor matrix (N x N)
    """
    N = len(states)
    matrix = numpy.zeros((N, N))
    actual_observation = errorize(sense(location, states, moves), error)
    for i in range(N):
        true_observation = sense(states[i], states, moves)
        d = 0
        for true_direction, actual_direction in zip(true_observation, actual_observation):
            if true_direction != actual_direction:
                d = d + 1
        matrix[i][i] = ((1 - error) ** (len(true_observation) - d)) * (error ** d)
    return matrix


def transition_model(states, moves):
    """
    Gets the transitional model.
    :param states: possible states
    :param moves: possible moves
    :return: transition matrix (N x N)
    """
    N = len(states)
    matrix = numpy.ndarray((N, N))
    for i in range(N):
        observation = sense(states[i], states, moves)
        n_zeroes = len(observation) - sum(observation)
        for j in range(N):
            matrix[i][j] = 0.0 if n_zeroes == 0 else 1.0 / n_zeroes
    return matrix


def viterbi(locations, states, moves, error):
    """
    Viterbi algorithm.
    :param locations: path locations
    :param states: possible states
    :param moves: possible moves
    :param error: error probability
    :return: best possible states
    """
    N = len(states)
    T = len(locations)

    M = numpy.ndarray((N, T + 1))
    M_star = numpy.ndarray((N, T + 1), numpy.int)

    transition_matrix = transition_model(states, moves)
    initial_transition = numpy.full((N,), 1.0 / N)
    for i in range(N):
        M[i][1] = initial_transition[i]
        M_star[i][1] = 0

    for t in range(2, T + 1):
        sensor_matrix = sensor_model(locations[t - 1], states, moves, error)
        for i in range(N):
            max_finding = []
            for j in range(N):
                max_finding.append(M[j][t - 1] * transition_matrix[j][i] * sensor_matrix[i][i])
            M[i][t] = max(max_finding)
            M_star[i][t] = numpy.argmax(max_finding)

    best_path = numpy.ndarray((T + 1,), numpy.int)
    max_finding = []
    for i in range(N):
        max_finding.append(M[i][T])
    best_path[T] = numpy.argmax(max_finding)
    for i in range(T, 1, -1):
        best_path[i - 1] = M_star[best_path[i]][i]
    return [states[index] for index in best_path[1:]]


def question_two(locations, states, moves, error):
    """
    Solution to question two.
    :param locations: path locations
    :param states: possible states
    :param moves: possible moves
    :param error: error probability
    :return: path accuracies at every time
    """
    viterbi_states = viterbi(locations, states, moves, error)
    accuracies = []
    accurate_count = 0
    for index, (location, viterbi_state) in enumerate(zip(locations, viterbi_states)):
        if location == viterbi_state:
            accurate_count = accurate_count + 1
        accuracies.append(accurate_count / (index + 1))
    return accuracies


def manhattan(location1, location2):
    """
    Manhattan distance.
    :param location1: x
    :param location2: y
    :return: distance
    """
    return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])


def forward(locations, states, moves, error):
    """
    Forward algorithm.
    :param locations: path locations
    :param states: possible states
    :param moves: possible moves
    :param error: error probability
    :return: best possible states
    """
    N = len(states)
    forward_states = []
    transition_matrix = transition_model(states, moves)
    forward_variable = numpy.full((N,), 1.0 / N)
    forward_states.append(states[numpy.argmax(forward_variable)])
    for location in locations[:-1]:
        sensor_matrix = sensor_model(location, states, moves, error)
        forward_variable = numpy.matmul(sensor_matrix, numpy.matmul(transition_matrix.T, forward_variable))
        if numpy.sum(forward_variable) != 0.0:
            forward_variable = forward_variable / numpy.sum(forward_variable)
        forward_states.append(states[numpy.argmax(forward_variable)])
    return forward_states


def question_one(locations, states, moves, error):
    """
    Solution to question one.
    :param locations: path locations
    :param states: possible states
    :param moves: possible moves
    :param error: error probability
    :return: distances
    """
    forward_states = forward(locations, states, moves, error)
    distances = []
    for location, forward_state in zip(locations, forward_states):
        distances.append(manhattan(location, forward_state))
    return distances


def sense(location, states, moves):
    """
    Sense the environment.
    :param location: current location
    :param states: possible states
    :param moves: possible moves
    :return: sensor output
    """
    ideal = [1] * len(moves)
    for index, move in enumerate(moves):
        if move(location) in states:
            ideal[index] = 0
    return ideal


def traverse(start, states, moves, n_steps):
    """
    Create a random traversal.
    :param start: start location
    :param states: possible states
    :param moves: possible moves
    :param n_steps: number of steps to take
    :return: path locations
    """
    locations = [start]
    for _ in range(n_steps - 1):
        observation = sense(locations[-1], states, moves)
        n_zeroes = len(observation) - sum(observation)
        if n_zeroes == 0:
            locations.append(locations[-1])
        else:
            random_direction = numpy.random.randint(0, n_zeroes)
            direction = [index for index, value in enumerate(observation) if value == 0][random_direction]
            locations.append(moves[direction](locations[-1]))
    return locations


def main(environment, moves, errors, n_steps, n_runs):
    """
    Handles input and plots the graphs.
    :param environment: environment as a matrix of obstacles
    :param moves: possible moves
    :param errors: possible error values
    :param n_steps: number of steps in traversal
    :param n_runs: number of runs to take average
    """
    states = []
    for i in range(len(environment)):
        for j in range(len(environment[i])):
            if not environment[i][j]:
                states.append((i, j))

    for error in errors:
        average_manhattan_distances = numpy.zeros(n_steps)
        average_path_accuracies = numpy.zeros(n_steps)
        for run in range(n_runs):
            start = states[numpy.random.randint(0, len(states))]
            locations = traverse(start, states, moves, n_steps)
            manhattan_distances = question_one(locations, states, moves, error)
            average_manhattan_distances = average_manhattan_distances + manhattan_distances
            path_accuracies = question_two(locations, states, moves, error)
            average_path_accuracies = average_path_accuracies + path_accuracies
        average_manhattan_distances = average_manhattan_distances / n_runs
        average_path_accuracies = average_path_accuracies / n_runs

        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.plot(average_manhattan_distances, label=str(error))
        matplotlib.pyplot.figure(2)
        matplotlib.pyplot.plot(average_path_accuracies, label=str(error))
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.title("Manhattan distances for error rates")
    matplotlib.pyplot.xlabel("Path length")
    matplotlib.pyplot.ylabel("Manhattan distance")
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.figure(2)
    matplotlib.pyplot.title("Path accuracies for error rates")
    matplotlib.pyplot.xlabel("Path length")
    matplotlib.pyplot.ylabel("Path accuracy")
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.show()


if __name__ == '__main__':
    numpy.random.seed(0)
    environment = [
        [False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False],
        [True, True, False, False, True, False, True, True, False, True, False, True, False, True, True, True],
        [True, False, False, False, True, False, True, False, False, False, False, False, False, True, True, False],
        [False, False, True, False, False, False, True, False, False, False, False, True, False, False, False, False]]
    moves = [lambda l: (l[0] - 1, l[1]), lambda l: (l[0], l[1] + 1),
             lambda l: (l[0], l[1] - 1), lambda l: (l[0] + 1, l[1])]
    errors = [0.0, 0.02, 0.05, 0.10, 0.20]
    n_runs = 500
    n_steps = 50
    main(environment, moves, errors, n_steps, n_runs)
