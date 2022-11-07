# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os
import scipy
import scipy.sparse
import scipy.sparse.linalg


# MRG packages
import _env


def is_in_interior_domain(node):
    """
    :param node: node of the grid
    :type node: int64
    :return: boolean
    :rtype: boolean
    """

    if node == _env.NODE_INTERIOR:
        return True
    else:
        return False  # ..warning: can be on the frontier


def is_on_dirichlet_boundary(node):
    """
    :param node: node of the grid
    :type node: int64
    :return: boolean
    :rtype: boolean
    """

    if node == _env.NODE_DIRICHLET:
        return True
    else:
        return False


def is_on_neumann_boundary(node):
    """
    :param node: node of the grid
    :type node: int64
    :return: boolean
    :rtype: boolean
    """

    if node == _env.NODE_NEUMANN:
        return True
    else:
        return False


def is_on_robin_boundary(node):
    """
    :param node: node of the grid
    :type node: int64
    :return: boolean
    :rtype: boolean
    """

    if node == _env.NODE_ROBIN:
        return True
    else:
        return False


def is_on_liner_a_boundary(node):
    """
    :param node: node of the grid
    :type node: int64
    :return: boolean
    :rtype: boolean
    """

    if node == _env.NODE_LINER_A:
        return True
    else:
        return False


def is_on_liner_b_boundary(node):
    """
    :param node: node of the grid
    :type node: int64
    :return: boolean
    :rtype: boolean
    """

    if node == _env.NODE_LINER_B:
        return True
    else:
        return False


def compute_stiffness_matrix(domain, space_step, f, beta_pde):
    """
    This function generates the stiffness matrix.

    :param domain:
    :param space_step:
    :param f:
    :param beta_pde:
    :return:
    """

    h = space_step
    (M, N) = numpy.shape(domain)
    K = M * N
    mat = scipy.sparse.lil_matrix((K, K), dtype=numpy.complex128)
    rhs = numpy.zeros((K, 1), dtype=numpy.complex128)
    for i in range(0, M):
        for j in range(0, N):
            row = i * N + j
            if is_in_interior_domain(domain[i, j]):
                mat[row, row] = -4.0 * beta_pde[i, j]
                mat[row, row - 1] = 1.0 * beta_pde[i, j - 1]
                mat[row, row + 1] = 1.0 * beta_pde[i, j + 1]
                mat[row, row + N] = 1.0 * beta_pde[i + 1, j]
                mat[row, row - N] = 1.0 * beta_pde[i - 1, j]
                rhs[row] = h ** 2 * f[i, j]
            else:
                mat[row, row] = 1.0
                rhs[row] = 0.0

    return mat, rhs


def compute_mass_matrix(domain, space_step, alpha_pde):
    """
    This function generates the mass matrix.

    :param domain:
    :param space_step:
    :param alpha_pde:
    :return:
    """

    h = space_step
    (M, N) = numpy.shape(domain)
    K = M * N
    mat = scipy.sparse.lil_matrix((K, K), dtype=numpy.complex128)
    rhs = numpy.zeros((K, 1), dtype=numpy.complex128)
    for i in range(0, M):
        for j in range(0, N):
            row = i * N + j
            if is_in_interior_domain(domain[i, j]):
                mat[row, row] = 1.0 * h ** 2 * alpha_pde[i, j]

    return mat, rhs


def compute_vgradu_matrix(domain, space_step, v):
    h = space_step
    (M, N) = numpy.shape(domain)
    K = M * N
    mat = scipy.sparse.lil_matrix((K, K), dtype=numpy.complex128)
    rhs = numpy.zeros((K, 1), dtype=numpy.complex128)
    for i in range(0, M):
        for j in range(0, N):
            row = i * N + j
            if is_in_interior_domain(domain[i, j]):
                mat[row, row] = - 2.0 * h * (v[i, j, 0] + v[i, j, 1])
                mat[row, row + 1] = h * v[i, j, 0]
                mat[row, row + N] = h * v[i, j, 1]
                if not is_in_interior_domain(domain[i + 1, j]):
                    mat[row, row] = 2.0 * h * (v[i, j, 0] - v[i, j, 1])
                    mat[row, row + 1] = 0.0
                    mat[row, row - 1] = -h * v[i, j, 0]
                if not is_in_interior_domain(domain[i, j + 1]):
                    mat[row, row] = 2.0 * h * (-v[i, j, 0] + v[i, j, 1])
                    mat[row, row + N] = 0.0
                    mat[row, row - N] = -h * v[i, j, 1]
            else:
                pass

    return mat, rhs


def compute_dirichlet_condition(domain, f_dir, alpha_dir, beta_pde, mat, rhs):
    """
    This function generate the dirichlet boundary condition.

    :param domain:
    :param g:
    :param alpha_dir:
    :param beta_pde:
    :param mat:
    :param rhs:
    :return:
    """

    (M, N) = numpy.shape(domain)
    for i in range(0, M):
        for j in range(0, N):
            # the interior domain AND the frontiers have to be strictly inside of domain
            if is_in_interior_domain(domain[i, j]):
                north = is_on_dirichlet_boundary(domain[i - 1, j])
                south = is_on_dirichlet_boundary(domain[i + 1, j])
                east = is_on_dirichlet_boundary(domain[i, j + 1])
                west = is_on_dirichlet_boundary(domain[i, j - 1])
                row = i * N + j
                if north:
                    mat[row, row - N] -= beta_pde[i - 1, j]
                    rhs[row] += -f_dir[i - 1, j] / alpha_dir[i - 1, j]
                if south:
                    mat[row, row + N] -= beta_pde[i + 1, j]
                    rhs[row] += -f_dir[i + 1, j] / alpha_dir[i + 1, j]
                if east:
                    mat[row, row + 1] -= beta_pde[i, j + 1]
                    rhs[row] += -f_dir[i, j + 1] / alpha_dir[i, j + 1]
                if west:
                    mat[row, row - 1] -= beta_pde[i, j - 1]
                    rhs[row] += -f_dir[i, j - 1] / alpha_dir[i, j - 1]

    return mat, rhs


def compute_neumann_condition(domain, space_step, f_neu, beta_neu, beta_pde, mat, rhs):
    """
    This function generate the neumann boundary condition.

    :param domain:
    :param space_step:
    :param s_N:
    :param beta_neu:
    :param beta_pde:
    :param mat:
    :param rhs:
    :return:
    """

    h = space_step
    (M, N) = numpy.shape(domain)
    for i in range(0, M):
        for j in range(0, N):
            # the interior domain AND the frontiers have to be strictly inside of domain
            if is_in_interior_domain(domain[i, j]):
                north = is_on_neumann_boundary(domain[i - 1, j])
                south = is_on_neumann_boundary(domain[i + 1, j])
                east = is_on_neumann_boundary(domain[i, j + 1])
                west = is_on_neumann_boundary(domain[i, j - 1])
                row = i * N + j
                if north:
                    mat[row, row - N] -= beta_pde[i - 1, j]
                    mat[row, row] += beta_pde[i, j]
                    rhs[row] += h * f_neu[i - 1, j] / beta_neu[i - 1, j]
                if south:
                    mat[row, row + N] -= beta_pde[i + 1, j]
                    mat[row, row] += beta_pde[i, j]
                    rhs[row] += h * f_neu[i + 1, j] / beta_neu[i + 1, j]
                if east:
                    mat[row, row + 1] -= beta_pde[i, j + 1]
                    mat[row, row] += beta_pde[i, j]
                    rhs[row] += h * f_neu[i, j + 1] / beta_neu[i, j + 1]
                if west:
                    mat[row, row - 1] -= beta_pde[i, j - 1]
                    mat[row, row] += beta_pde[i, j]
                    rhs[row] += h * f_neu[i, j - 1] / beta_neu[i, j - 1]

    return mat, rhs


def compute_robin_condition(domain, space_step, f_rob, alpha_rob, beta_rob, beta_pde, mat, rhs):
    """
    This function generate the robin boundary condition.

    :param domain:
    :param space_step:
    :param f_rob:
    :param alpha_rob:
    :param beta_rob:
    :param beta_pde:
    :param mat:
    :param rhs:
    :return:
    """

    h = space_step
    (M, N) = numpy.shape(domain)
    for i in range(0, M):
        for j in range(0, N):
            # the interior domain AND the frontiers have to be strictly inside of domain
            if is_in_interior_domain(domain[i, j]):
                north = is_on_robin_boundary(domain[i - 1, j])
                south = is_on_robin_boundary(domain[i + 1, j])
                east = is_on_robin_boundary(domain[i, j + 1])
                west = is_on_robin_boundary(domain[i, j - 1])
                row = i * N + j
                if north:
                    mat[row, row - N] -= beta_pde[i - 1, j]
                    mat[row, row] += beta_pde[i, j] + (alpha_rob[i - 1, j] / beta_pde[i, j]) * h / (
                        beta_rob[i - 1, j] - (alpha_rob[i - 1, j] / beta_pde[i, j] * h))
                    rhs[row] += (h / (beta_rob[i - 1, j] - ((alpha_rob[i - 1, j] / beta_pde[i, j]) * h))) * f_rob[
                        i - 1, j]
                if south:
                    mat[row, row + N] -= beta_pde[i + 1, j]
                    mat[row, row] += beta_pde[i, j] + (alpha_rob[i + 1, j] / beta_pde[i, j]) * h / (
                        beta_rob[i + 1, j] - (alpha_rob[i + 1, j] / beta_pde[i, j] * h))
                    rhs[row] += (h / (beta_rob[i + 1, j] - ((alpha_rob[i + 1, j] / beta_pde[i, j]) * h))) * f_rob[
                        i + 1, j]
                if east:
                    mat[row, row + 1] -= beta_pde[i, j + 1]
                    mat[row, row] += beta_pde[i, j] + (alpha_rob[i, j + 1] / beta_pde[i, j]) * h / (
                        beta_rob[i, j + 1] - (alpha_rob[i, j + 1] / beta_pde[i, j] * h))
                    rhs[row] += (h / (beta_rob[i, j + 1] - ((alpha_rob[i, j + 1] / beta_pde[i, j]) * h))) * f_rob[
                        i, j + 1]
                if west:
                    mat[row, row - 1] -= beta_pde[i, j - 1]
                    mat[row, row] += beta_pde[i, j] + (alpha_rob[i, j - 1] / beta_pde[i, j]) * h / (
                        beta_rob[i, j - 1] - (alpha_rob[i, j - 1] / beta_pde[i, j] * h))
                    rhs[row] += (h / (beta_rob[i, j - 1] - ((alpha_rob[i, j - 1] / beta_pde[i, j]) * h))) * f_rob[
                        i, j - 1]

    return mat, rhs


def solve_helmholtz(domain, space_step, omega,
                    f, f_dir, f_neu, f_rob,
                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob):
    """
    :param domain:
    :param space_step:
    :param omega:
    :param f:
    :param f_dir:
    :param f_neu:
    :param f_rob:
    :param beta_pde:
    :param alpha_pde:
    :param alpha_dir:
    :param beta_neu:
    :param beta_rob:
    :param alpha_rob:
    :return:
    """

    (M, N) = numpy.shape(domain)
    # -- create stiffness matrix
    mat_temp, rhs_temp = compute_stiffness_matrix(
        domain, space_step, f, beta_pde)
    mat = mat_temp
    rhs = rhs_temp
    # -- create mass matrix
    mat_temp, rhs_temp = compute_mass_matrix(domain, space_step, alpha_pde)
    mat = mat + omega ** 2 * mat_temp
    rhs = rhs
    # -- create convection (vgradu) matrix
    # <!--
    # v = numpy.zeros((M, N, 2), dtype=numpy.float64)
    # xc = int(M//4) * space_step
    # yc = int(N//2) * space_step
    # for i in range(0, M):
    #     for j in range(0, N):
    #         xi = int(i) * space_step
    #         yi = int(j) * space_step
    #         import math
    #         r = math.sqrt( (xi-xc)**2 + (yi-yc)**2 )
    #         if (xi-xc) == 0:
    #             theta = 0.0
    #         elif (yi-yc) / (xi-xc) < 0:
    #             theta = math.atan( -(yi-yc) / (xi-xc) )
    #         elif (yi-yc) / (xi-xc) > 0:
    #             theta = math.atan( (yi - yc) / (xi - xc))
    #
    #         if r == 0.0:
    #             r = 1.0
    #         costheta = (xi - xc) / r
    #         sintheta = (yi - yc) / r
    #         v[i, j, 0] = costheta
    #         v[i, j, 1] = sintheta
    #
    # xploti, yplotj = numpy.meshgrid(numpy.arange(0, N, 1), numpy.arange(0, M, 1))
    # print(xploti.shape[:], yplotj.shape[:])
    # print(v.shape[:])
    #
    # import matplotlib.pyplot
    # #matplotlib.pyplot.quiver(xploti, yplotj, v[:,:,0], v[:,:,1])
    # fig, ax = matplotlib.pyplot.subplots()
    # ax.set_xlim(0, N)
    # ax.set_ylim(M, 0)
    # q = ax.quiver(xploti, yplotj, v[:,:,0], v[:,:,1], units='xy', scale=0.51, color='red')
    # ax.set_aspect('equal')
    # #matplotlib.pyplot.xlim(-5, 5)
    # #matplotlib.pyplot.ylim(-5, 5)
    # matplotlib.pyplot.show()
    #
    # #exit(-6)
    #
    # mat_temp, rhs_temp = compute_vgradu_matrix(domain, space_step, v)
    # mat = mat + mat_temp
    # rhs = rhs
    # -->

    # -- set dirichlet boundary conditions
    mat, rhs = compute_dirichlet_condition(
        domain, f_dir, alpha_dir, beta_pde, mat, rhs)
    # -- set neumann boundary conditions boundary conditions
    mat, rhs = compute_neumann_condition(
        domain, space_step, f_neu, beta_neu, beta_pde, mat, rhs)
    # -- set robin boundary conditions
    mat, rhs = compute_robin_condition(
        domain, space_step, f_rob, alpha_rob, beta_rob, beta_pde, mat, rhs)

    # print(mat.shape[:])

    # -- solve linear system
    sol = scipy.sparse.linalg.spsolve(mat, rhs)

    u = numpy.zeros((M, N), dtype=numpy.complex128)
    # -- map solution to the grid
    # for k in range(M * N):
    #     i = k // N
    #     j = k % N
    #     u[i, j] = sol[k]

    for i in range(0, M):
        for j in range(0, N):
            row = i * N + j
            if is_in_interior_domain(domain[i, j]) or is_on_robin_boundary(domain[i, j]):
                u[i, j] = sol[row]

    return u


def compute_robin_condition_down(domain, u, space_step, beta_rob, alpha_rob):
    """

    :param domain:
    :param u:
    :param space_step:
    :param beta_rob:
    :param alpha_rob:
    :return:
    """

    h = space_step
    (M, N) = numpy.shape(domain)
    value = numpy.zeros((M, N), dtype=numpy.complex128)
    for i in range(0, M):
        for j in range(0, N):
            if is_in_interior_domain(domain[i, j]):
                north = is_on_robin_boundary(domain[i - 1, j])
                south = is_on_robin_boundary(domain[i + 1, j])
                east = is_on_robin_boundary(domain[i, j + 1])
                west = is_on_robin_boundary(domain[i, j - 1])
                if north:
                    value[i - 1, j] += beta_rob[i - 1, j] * (u[i, j] - u[i - 1, j]) / h + alpha_rob[
                        i - 1, j] * (u[i - 1, j] + u[i, j]) / 2
                if south:
                    value[i + 1, j] += beta_rob[i + 1, j] * (u[i, j] - u[i + 1, j]) / h + alpha_rob[
                        i + 1, j] * (u[i + 1, j] + u[i, j]) / 2
                if east:
                    value[i, j + 1] += beta_rob[i, j + 1] * (u[i, j] - u[i, j + 1]) / h + alpha_rob[
                        i, j + 1] * (u[i, j + 1] + u[i, j]) / 2
                if west:
                    value[i, j - 1] += beta_rob[i, j - 1] * (u[i, j] - u[i, j - 1]) / h + alpha_rob[
                        i, j - 1] * (u[i, j - 1] + u[i, j]) / 2

    return value


def compute_robin_condition_up(domain, u, space_step, beta_rob, alpha_rob):
    """

    :param domain:
    :param u:
    :param space_step:
    :param beta_rob:
    :param alpha_rob:
    :return:
    """

    h = space_step
    (M, N) = numpy.shape(domain)
    value = numpy.zeros((M, N), dtype=numpy.complex128)
    # for i in range(M - 1, -1, -1):  # la frontière est sur un bord donc on ne regarde que
    # les points qui sont strictement à l'intérieur (i.e d'où le 2:M-2 et 2:N-2)
    #    for j in range(N - 1, -1, -1):
    for i in range(0, M):
        for j in range(0, N):
            if is_in_interior_domain(domain[i, j]):
                north = is_on_robin_boundary(domain[i - 1, j])
                south = is_on_robin_boundary(domain[i + 1, j])
                east = is_on_robin_boundary(domain[i, j + 1])
                west = is_on_robin_boundary(domain[i, j - 1])
                if north:
                    value[i - 1, j] += beta_rob[i - 1, j] * (u[i, j] - u[i - 1, j]) / h + alpha_rob[
                        i - 1, j] * (u[i - 1, j] + u[i, j]) / 2
                if south:
                    value[i + 1, j] += beta_rob[i + 1, j] * (u[i, j] - u[i + 1, j]) / h + alpha_rob[
                        i + 1, j] * (u[i + 1, j] + u[i, j]) / 2
                if east:
                    value[i, j + 1] += beta_rob[i, j + 1] * (u[i, j] - u[i, j + 1]) / h + alpha_rob[
                        i, j + 1] * (u[i, j + 1] + u[i, j]) / 2
                if west:
                    value[i, j - 1] += beta_rob[i, j - 1] * (u[i, j] - u[i, j - 1]) / h + alpha_rob[
                        i, j - 1] * (u[i, j - 1] + u[i, j]) / 2

    return value


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
