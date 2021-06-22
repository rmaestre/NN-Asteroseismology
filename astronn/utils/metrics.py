import logging

import numpy as np
import math

dnu_sun = 134.8
rho_sun = 1.409


def get_log_l(P, A=1.4722, e_A=0.1, B=2.6089, e_B=0.096):
    """
    P-L relation for McNamara (2011)
    """
    return A * np.log10(P) + B


def get_P(log_L, A=1.4722, e_A=0.1, B=2.6089, e_B=0.096):
    """
    P variable isolated from log_L relation
    """
    return np.power(10, (log_L - B) / (A))


def get_rho_from_P(P, Q=0.033):
    """
    Get rho from P
    """
    return np.power(Q / P, 2)


def p_error(log_L, A=1.4722, e_A=0.1, B=2.6089, e_B=0.096):
    """
    Rabge of error derived from the PL relation
    """
    return np.sqrt(
        np.power(
            np.power(10, (log_L - B) / A) * (2.30258509 / np.power(A, 2)) * log_L * e_A,
            2,
        )
        + np.power(np.power(10, (log_L - B) / A) * (2.30258509 / A) * e_B, 2)
    )


def get_rho(delta_nu):
    """
    Relation Rodriguez-Martin et.al. 2020
    """
    return 1.6 * np.power(delta_nu, 2.02)


def rho_error(delta_nu, A=1.6, e_A=0.5, B=2.02, e_B=0.1):
    """
    Error of relation Rodriguez-Martin et.al. 2020
    """
    return np.sqrt(
        np.power(np.power(delta_nu, B) * e_A, 2)
        + np.power(A * np.power(delta_nu, B) * np.log10(delta_nu) * e_B, 2)
    )


def get_dnu_from_rho(rho):
    """
    """
    return math.pow(rho / 1.6, math.pow(2.02, -1))


def dnu_error(rho, A=1.6, e_A=0.0, B=2.02, e_B=0.1):
    """
    """
    return np.sqrt(
        math.pow(
            -math.pow(rho / A, 1 / B - 1) * (1 / B / math.pow(A, 2)) * rho * e_A, 2
        )
        + math.pow(
            -math.pow(rho / A, 1 / B)
            / math.pow(B, 2)
            * math.log(rho / A, math.e)
            * e_B,
            2,
        )
    )
