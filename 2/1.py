import matplotlib.pyplot
import numpy


def metropolis_hastings(x_0, q, p, i):
    """
    The Metropolis Hastings Algorithm.
    :param x_0: start state
    :param q: proposal function
    :param p: probability function
    :param i: iterations
    :return: list of states
    """
    x = [x_0]
    for _ in range(i):
        x_t = x[-1]
        x_c = q(x_t)
        a = numpy.min([1.0, (q(x_c) * p(x_c)) / (q(x_t) * p(x_t))])
        u = numpy.random.uniform()
        x_t_1 = x_c if u < a else x_t
        x.append(x_t_1)
    return x


def p(x):
    """
    Given function.
    :rtype: probability
    """
    return numpy.exp(-numpy.power(x, 4.0)) * (2.0 + numpy.sin(5.0 * x) + numpy.sin(-2.0 * numpy.power(x, 2.0)))


def n(mu, sigma):
    """
    Normal distribution proposal.
    :param mu: mean
    :param sigma: standard deviation
    :return: value
    """
    return numpy.random.normal(mu, sigma)


def main():
    """
    Call functions and draws plots.
    """
    xs = []
    sigmas = (0.05, 1.0, 50.0)
    for sigma in sigmas:
        xs.append(metropolis_hastings(-1.0, lambda mu: (n(mu, sigma)), p, 1500))

    figure = matplotlib.pyplot.figure(1)
    figure.suptitle('Samples vs iteration')
    for i in range(len(xs)):
        subplot = matplotlib.pyplot.subplot(len(xs), 1, i + 1)
        subplot.set_title('$\sigma=$' + str(sigmas[i]))
        matplotlib.pyplot.plot(xs[i])

    figure = matplotlib.pyplot.figure(2)
    figure.suptitle('Histogram of samples')
    for i in range(len(xs)):
        subplot = matplotlib.pyplot.subplot(len(xs), 1, i + 1)
        subplot.set_title('$\sigma=$' + str(sigmas[i]))
        matplotlib.pyplot.hist(xs[i], bins=100, density=True)

    matplotlib.pyplot.show()


if __name__ == '__main__':
    main()
