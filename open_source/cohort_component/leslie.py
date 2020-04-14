"""
This defines a Leslie matrix capable of computing its inherent
growth rate and performing efficient incrementing of past population.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np


LOGGER = logging.getLogger("fbd_research.cohort_component.leslie")


def has_dims(numpy_array, dim_str):
    """
    Verifies the count of the dimensions but let's you proclaim what they are.
    Used as::

       assert has_dims(survivorship, "year sex age")

    Args:
        numpy_array (np.ndarray):
        dim_str (str):

    Returns:
        bool
    """
    return len(numpy_array.shape) == len(dim_str.split())


class LeslieMatrix:
    """
    The Leslie matrix here assembles fertility and survivorship.
    This version applies survivorship to both sexes.
    Its internal representation is two vectors, not a matrix.
    """
    def __init__(self, fertility, fertility_begin, survivorship, srb=1.05):
        """
        For a population with :math:`n` entries, survivorship has
        :math:`n+1` entries, where the first is survival of newborns
        and the last accounts for the last age interval being an open
        interval.

        Args:
            fertility (np.ndarray): Children born per female.
            fertility_begin (int):  First fertile age group.
            survivorship (np.ndarray): Probability a population survives.
            srb (float): Sex ratio, usually 1.05.
        """
        assert has_dims(fertility, "age")
        assert has_dims(survivorship, "sex age")

        fb, fe = (fertility_begin, fertility_begin + fertility.shape[-1])
        self._fertile_limits = (fb, fe)
        # fb=1, fe=3, surv.shape=6
        self._survivorship = survivorship
        # Read this from PHG, section 6.4 _Projections in Matrix Notation._
        # Survival of newborns is by sex.
        sex_ratio = np.array([[srb / (1 + srb)], [1 / (1 + srb)]])
        self._k = 0.5 * survivorship[:, :1] * sex_ratio
        # Fertility is only women. 0-based index of 1 is sex_id=2.
        self._fertile = np.concatenate([
            np.zeros((fb - 1,), dtype=np.double),
            [fertility[0] * survivorship[1, fb]],
            (fertility[:-1] + fertility[1:] * survivorship[1, fb + 1:fe]),
            [fertility[-1]],
            np.zeros((survivorship.shape[-1] - fe - 1), dtype=np.double)
        ], axis=0)
        assert self._fertile.shape[-1] == survivorship.shape[-1] - 1

    def increment(self, population):
        """
        Given population at time :math:`t`, return population at :math:`t+1`.
        This does a matrix multiply without making the matrix with which
        to multiply.
        """
        # off_diagonal includes lower, right-hand corner
        off_diagonal = self._survivorship[:, 1:] * population
        pop_new = np.concatenate([
            self._k * np.tile(np.inner(self._fertile, population[1]), (2, 1)),
            off_diagonal[:, :-2],
            off_diagonal[:, -2:-1] + off_diagonal[:, -1:]
        ], axis=1)
        return pop_new

    def eigensystem(self):
        """
        Eigensystem of the fertile ages. This is what determines growth.
        """
        values, vectors = np.linalg.eig(self._irreducible_matrix())
        return values, vectors

    def svd(self):
        """Singular value decomposition of the reducible
        submatrix containing all fertile ages. This returns
        numpy's svd, so that all eigenvalues are there. The leading
        eigenvalue is the growth rate of the matrix. There will be only
        one positive eigenvalue, and it will be real."""
        u, sigma, v = np.linalg.svd(self._irreducible_matrix())
        return (u, sigma, v)

    def _irreducible_matrix(self):
        # Infertile ages don't affect population size and make matrix stiff.
        end = self._fertile_limits[1]
        return self._matrix()[:end, :end]

    def _matrix(self):
        # Just female.
        leslie = np.roll(np.diag(self._survivorship[1, 1:]),
                         1, 0)
        leslie[-1, -1] = self._survivorship[1, -1]
        leslie[0] = self._fertile
        return leslie
