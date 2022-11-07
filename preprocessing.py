# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os
from processing import is_on_robin_boundary

# MRG packages
import _env


def set2zero(alpha, domain_omega):
    """This function is useful during the optimisation procedure. It makes sure
    that the density is null everywhere except on the Robin frontier

    Parameter:
        alpha: Matrix (MxN, dtype=complex), this matrix is the density multiply
        by the coefficient of absorbtion;
        domain_omega: Matrix (MxN), it defines the domain and the shape of the
        Robin frontier.
    """

    (M, N) = numpy.shape(domain_omega)
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] != _env.NODE_ROBIN:
                alpha[i, j] = 0

    return alpha


def replace_old_by_new(u, oldvalue, newvalue):
    (M, N) = numpy.shape(u)
    for i in range(M):
        for j in range(N):
            if u[i, j] == oldvalue:
                u[i, j] = newvalue
    return u


def _set_coefficients_of_pde(M, N):
    beta_pde = numpy.ones((M, N), dtype=numpy.complex128)
    alpha_pde = numpy.ones((M, N), dtype=numpy.complex128)
    alpha_dir = numpy.ones((M, N), dtype=numpy.complex128)
    beta_neu = numpy.ones((M, N), dtype=numpy.complex128)
    alpha_rob = numpy.ones((M, N), dtype=numpy.complex128)
    beta_rob = numpy.ones((M, N), dtype=numpy.complex128)
    return beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob


def _set_rhs_of_pde(M, N):
    f = numpy.zeros((M, N), dtype=numpy.complex128)
    f_dir = numpy.zeros((M, N), dtype=numpy.complex128)
    f_neu = numpy.zeros((M, N), dtype=numpy.complex128)
    f_rob = numpy.zeros((M, N), dtype=numpy.complex128)
    return f, f_dir, f_neu, f_rob


def _set_geometry_of_domain(M, N, level=0):
    # -- define geometry
    domain_omega = numpy.zeros((M, N), dtype=numpy.int64)
    # # attention version pour les fractales
    # domain_omega[0:M, 0:N] = _env.NODE_INTERIOR
    # domain_omega[0, 0:N] = _env.NODE_DIRICHLET  # north
    # domain_omega[M-1, 0:N] = _env.NODE_ROBIN  # south
    # domain_omega[0:M, 0] = _env.NODE_NEUMANN  # west
    # domain_omega[0:M, N-1] = _env.NODE_NEUMANN  # east

    # attention version avec/sans les fractales
    domain_omega[0:M, 0:N] = _env.NODE_INTERIOR  # interior
    domain_omega[0, 0:N] = _env.NODE_DIRICHLET  # north
    domain_omega[M-1, 0:N] = _env.NODE_NEUMANN  # south
    domain_omega[0:M, 0] = _env.NODE_NEUMANN  # west
    domain_omega[0:M, N - 1] = _env.NODE_NEUMANN  # east

    if level == 0:
        domain_omega[N, 0:N] = _env.NODE_ROBIN  # south
    else:
        domain_omega[M-1, 0:N] = _env.NODE_NEUMANN  # south

    # -- define fractal
    nodes = create_fractal_nodes(
        [numpy.array([[0], [N]]), numpy.array([[N], [N]])], level)
    x, y = create_fractal_coordinates(nodes, domain_omega)
    x_plot = numpy.zeros(len(x) - 1, dtype=numpy.float64)
    y_plot = numpy.zeros(len(x) - 1, dtype=numpy.float64)

    # -- plot fractal
    #matplotlib.pyplot.plot(x, y)
    # matplotlib.pyplot.title('Fractal')
    # matplotlib.pyplot.show()

    # -- define smaller geometry to take into account the fractal as boundary condition
    for k in range(0, len(x) - 1):
        domain_omega[int(y[k]), int(x[k])] = _env.NODE_ROBIN
    seed1 = [M-2, N-2]
    domain_omega = partition_domain(domain_omega, seed1)

    return domain_omega, x, y, x_plot, y_plot


def _set_chi(M, N, x, y):
    chi = numpy.zeros((M, N), dtype=numpy.float64)
    k_begin = (len(x) - 1) // 5
    k_end = 3 * (len(x) - 1) // 5
    val = 1.0
    for k in range(k_begin, k_end):
        chi[int(y[k]), int(x[k])] = val
    return chi


def create_motif_koch(A, B):
    """
    Generate a Koch pattern.

    :param A: starting node of the pattern
    :param B: ending node of the pattern
    :type A: numpy.array((2,1), dtype=)
    :type B: numpy.array((2,1), dtype=)
    :return:
    :rtype:
    """

    alpha = numpy.pi / 2.0
    radius = numpy.array([[numpy.cos(alpha), -numpy.sin(alpha)],
                          [numpy.sin(alpha), numpy.cos(alpha)]])
    distance = (B - A) / 4.0
    C = A + distance
    D = C + numpy.dot(radius, distance)
    E = D + distance
    F = E - numpy.dot(radius, distance)
    G = F - numpy.dot(radius, distance)
    H = G + distance
    I = H + numpy.dot(radius, distance)

    return [A, C, D, E, F, G, H, I, B]


def create_fractal_nodes(nodes, n_iter):
    """
    Create the list of nodes on the fractal.

    :param nodes: list of nodes of the fractal
    :param n_iter: level of the fractal
    :type nodes: numpy.array((2,1), dtype=)
    :type n_iter: int64
    :return:
    :rtype:
    """

    n = 0
    while n < n_iter:
        new_nodes = []
        for k in range(len(nodes) - 1):
            a = nodes[k]
            b = nodes[k + 1]
            temp = create_motif_koch(a, b)
            new_nodes = new_nodes + temp
        n += 1
        nodes = new_nodes

    return nodes


def create_fractal_coordinates(nodes, domain):
    """
    This function transforms the list of nodes of the fractal into coordinates.

    :param nodes: list of nodes of the fractal
    :param n_iter: level of the fractal
    :type nodes: numpy.array((2,1), dtype=)
    :type n_iter: int64
    :return:
    :rtype:

    ..warning: nodes must be precomputed with function create_fractal_nodes
    """

    a = []
    b = []
    for k in range(0, len(nodes)):
        a.append(round(nodes[k][0][0]))
        b.append(round(nodes[k][1][0]))
    current_node = [[a[k], b[k]] for k in range(len(nodes))]
    l = []
    for k in range(0, len(current_node) - 1):
        n = nodes[k + 1][0] - nodes[k][0]
        m = nodes[k + 1][1] - nodes[k][1]
        if n > 0:
            for o in range(0, int(n) + 1):
                a = [[current_node[k][0] + o], [current_node[k][1]]]
                l.append(a)
        elif m > 0:
            for o in range(0, int(m) + 1):
                a = [[current_node[k][0]], [current_node[k][1] + o]]
                l.append(a)
        elif m < 0:
            for o in range(0, int(-m) + 1):
                a = [[current_node[k][0]], [current_node[k][1] - o]]
                l.append(a)
        elif n < 0:
            for o in range(0, int(-n) + 1):
                a = [[current_node[k][0] - o], [current_node[k][1]]]
                l.append(a)
    x = []
    y = []
    for k in range(0, len(l)):
        x.append(l[k][0][0])
        y.append(l[k][1][0])

    (M, N) = numpy.shape(domain)

    while x[-1] < N:
        x.append(x[-1] + 1)
        y.append(y[-1])

    while x[0] > 0:
        x.insert(0, x[0] - 1)
        y.insert(0, y[0])

    return x, y


def is_on_boundary(node):
    """
    :param node: node of the grid
    :type node: int64
    :return: boolean
    :rtype: boolean
    """

    if node not in [_env.NODE_ROBIN, _env.NODE_NEUMANN, _env.NODE_DIRICHLET]:
        return 'NOT BOUNDARY'
    else:
        return 'BOUNDARY'


def partition_domain(domain, seed):
    """

    :param domain:
    :param seed:
    :return:
    """

    (M, N) = numpy.shape(domain)
    Neighbours = [seed]
    Visited = []

    if is_on_boundary(domain[seed[0], seed[1]]) == 'BOUNDARY':
        return 'Error: choose another point for seed'

    else:
        count = 0
        while len(Neighbours) > 0:

            i = Neighbours[0][0]
            j = Neighbours[0][1]

            if i == 0:  # Il n'y a pas de nord
                if j == 0:  # Il n'y a pas d'ouest
                    south = [i + 1, j]
                    east = [i, j + 1]
                    a = 'Not Exist'
                    b = is_on_boundary(domain[south[0], south[1]])
                    c = is_on_boundary(domain[east[0], east[1]])
                    d = 'Not Exist'
                elif j == N - 1:  # Il n'y a pas d'est
                    south = [i + 1, j]
                    west = [i, j - 1]
                    a = 'Not Exist'
                    b = is_on_boundary(domain[south[0], south[1]])
                    c = 'Not Exist'
                    d = is_on_boundary(domain[west[0], west[1]])
                else:
                    south = [i + 1, j]
                    west = [i, j - 1]
                    east = [i, j + 1]
                    a = 'Not Exist'
                    b = is_on_boundary(domain[south[0], south[1]])
                    c = is_on_boundary(domain[east[0], east[1]])
                    d = is_on_boundary(domain[west[0], west[1]])
            elif i == M - 1:  # Il n'y a pas de sud
                if j == 0:  # Il n'y a pas d'ouest
                    north = [i - 1, j]
                    east = [i, j + 1]
                    a = is_on_boundary(domain[north[0], north[1]])
                    b = 'Not Exist'
                    c = is_on_boundary(domain[east[0], east[1]])
                    d = 'Not Exist'
                elif j == N - 1:  # Il n'y a pas d'est
                    north = [i - 1, j]
                    west = [i, j - 1]
                    a = is_on_boundary(domain[north[0], north[1]])
                    b = 'Not Exist'
                    c = 'Not Exist'
                    d = is_on_boundary(domain[west[0], west[1]])
                else:
                    north = [i - 1, j]
                    west = [i, j - 1]
                    east = [i, j + 1]
                    a = is_on_boundary(domain[north[0], north[1]])
                    b = 'Not Exist'
                    c = is_on_boundary(domain[east[0], east[1]])
                    d = is_on_boundary(domain[west[0], west[1]])
            elif j == N - 1:  # Il n'y a pas de est
                if i == 0:  # Il n'y a pas de nord
                    south = [i + 1, j]
                    west = [i, j - 1]
                    a = 'Not Exist'
                    b = is_on_boundary(domain[south[0], south[1]])
                    c = 'Not Exist'
                    d = is_on_boundary(domain[west[0], west[1]])
                elif i == N - 1:  # Il n'y a pas de sud
                    north = [i - 1, j]
                    west = [i, j - 1]
                    a = is_on_boundary(domain[north[0], north[1]])
                    b = 'Not Exist'
                    c = 'Not Exist'
                    d = is_on_boundary(domain[west[0], west[1]])
                else:
                    north = [i - 1, j]
                    south = [i + 1, j]
                    west = [i, j - 1]
                    a = is_on_boundary(domain[north[0], north[1]])
                    b = 'Not Exist'
                    c = is_on_boundary(domain[east[0], east[1]])
                    d = is_on_boundary(domain[west[0], west[1]])
            elif j == 0:  # Il n'y a pas d'ouest
                if i == 0:  # Il n'y a pas de nord
                    south = [i + 1, j]
                    east = [i, j + 1]
                    a = 'Not Exist'
                    b = is_on_boundary(domain[south[0], south[1]])
                    c = is_on_boundary(domain[east[0], east[1]])
                    d = 'Not Exist'
                elif i == N - 1:  # Il n'y a pas de sud
                    north = [i - 1, j]
                    east = [i, j + 1]
                    a = is_on_boundary(domain[north[0], north[1]])
                    b = 'Not Exist'
                    c = is_on_boundary(domain[east[0], east[1]])
                    d = 'Not Exist'
                else:
                    north = [i - 1, j]
                    south = [i + 1, j]
                    east = [i, j + 1]
                    a = is_on_boundary(domain[north[0], north[1]])
                    b = is_on_boundary(domain[south[0], south[1]])
                    c = is_on_boundary(domain[east[0], east[1]])
                    d = 'Not Exist'

            else:
                north = [i - 1, j]
                south = [i + 1, j]
                east = [i, j + 1]
                west = [i, j - 1]
                a = is_on_boundary(domain[north[0], north[1]])
                b = is_on_boundary(domain[south[0], south[1]])
                c = is_on_boundary(domain[east[0], east[1]])
                d = is_on_boundary(domain[west[0], west[1]])

            domain[i, j] = _env.NODE_COMPLEMENTARY
            Neighbours.pop(0)
            Visited.append([i, j])
            count += 1

            if a == 'NOT BOUNDARY' and north not in Visited and north not in Neighbours:
                Neighbours.append(north)
            if b == 'NOT BOUNDARY' and south not in Visited and south not in Neighbours:
                Neighbours.append(south)
            if c == 'NOT BOUNDARY' and east not in Visited and east not in Neighbours:
                Neighbours.append(east)
            if d == 'NOT BOUNDARY' and west not in Visited and west not in Neighbours:
                Neighbours.append(west)
            if count > M * N:
                return Neighbours, Visited

    return domain


def surlignefractale(domain, color_domain, color_seed):
    """Color the points in the interior domain at a distance to the boundary equal one."""

    (M, N) = numpy.shape(domain)

    for i in range(M):
        for j in range(N):
            if BelongsInteriorDomain(domain[i, j]) == color_domain:
                a = is_on_robin_boundary(domain[i - 1, j])
                b = is_on_robin_boundary(domain[i + 1, j])
                c = is_on_robin_boundary(domain[i, j - 1])
                d = is_on_robin_boundary(domain[i, j + 1])
                e = is_on_robin_boundary(domain[i + 1, j + 1])
                f = is_on_robin_boundary(domain[i - 1, j - 1])
                g = is_on_robin_boundary(domain[i + 1, j - 1])
                h = is_on_robin_boundary(domain[i - 1, j + 1])

                if a:
                    domain[i, j] = color_seed
                if b:
                    domain[i, j] = color_seed
                if c:
                    domain[i, j] = color_seed
                if d:
                    domain[i, j] = color_seed
                if e:
                    domain[i, j] = color_seed
                if f:
                    domain[i, j] = color_seed
                if g:
                    domain[i, j] = color_seed
                if h:
                    domain[i, j] = color_seed

    return domain
