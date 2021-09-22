import logging

import numpy as np
import math
import matplotlib.pyplot as plt

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


def get_rho_gh17(delta_nu):
    """
    Relation Rodriguez-Martin et.al. 2020
    """
    return 1.5 * np.power(delta_nu, 2.04)

def get_rho_gh17_upper_bound(delta_nu, e_A=0.5, e_B=0.0):
    """
    Relation Rodriguez-Martin et.al. 2020
    """
    return (1.5 + e_A) * np.power(delta_nu, (2.04 + e_B))

def get_rho_gh17_lower_bound(delta_nu, e_A=0.5, e_B=0.0):
    """
    Relation Rodriguez-Martin et.al. 2020
    """
    return (1.5 - (e_A - 0.1)) * np.power(delta_nu, (2.04 - e_B))

def get_rho(delta_nu):
    """
    Relation Rodriguez-Martin et.al. 2020
    """
    return 1.6 * np.power(delta_nu, 2.02)

def get_rho_upper_bound(delta_nu, e_A=0.5, e_B=0.0):
    """
    Relation Rodriguez-Martin et.al. 2020
    """
    return (1.6 + e_A) * np.power(delta_nu, (2.02 + e_B))


def get_rho_lower_bound(delta_nu, e_A=0.5, e_B=0.0):
    """
    Relation Rodriguez-Martin et.al. 2020
    """
    return (1.6 - (e_A - 0.1)) * np.power(delta_nu, (2.02 - e_B))


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
    return np.power(rho / 1.6, np.power(2.02, -1))


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


def echelle(frec_f, Dnu, num_freq=30, no_g=58, saveplot="Y"):
    # freqs = frequencies (in muHz)
    # Dnu = large separation (in muHz)
    # plot = variable to indicate if the function will plot the results
    # tol = 1 #Tolerance for the search of ridges
    # TOLERANCE SHOULD DEPEND ON DNU???????????????????????

    #    tol = Dnu*0.05 #5% of Dnu for the tolerance. See models of AGH's thesis
    tol = (
        Dnu * 0.015
    )  # Tolerance is around 1muHz for a Dnu~65muHz. See models from AGH's thesis and GH13

    # THIS IS THE PLOT OF THE ECHELLE DIAGRAM

    # We check if freqs is a finlename or a vector
    if type(frec_f) is np.ndarray:
        print(
            "Input is an array. ASSUMING ORDERED FREQUENCIES AND MICRO-Hz! Plotting échelle\n"
        )
        fre = frec_f  # Then, use the vector
    # If not, trying to reading the file
    elif os.path.isfile(frec_f):
        print("Input is a file. Reading file before plotting\n")
        fre, ampli = read(frec_f)
        fre = fre / 0.0864  # We need the frequencies in muHz
        # We need to have the frequencies ordered by amplitude
        s_ind = np.argsort(
            ampli
        )  # Indices for sorting the values. This is in ascending order
        s_ind = s_ind[::-1]  # Indices in descending order
        fre = fre[s_ind]  # Frequencies ordered by amplitude
    #        amp = ampli[s_ind] #Ordering amplitude
    else:
        sys.exit("The file " + frec_f + " does not exists\n")

    freqs = fre[fre > no_g]
    #    amp = ampli[fre > no_g]

    range_fre = np.size(freqs)
    #    print range_fre
    Dnu_d = Dnu * 0.0864  # Dnu in c/d

    mod_Dnu = freqs % Dnu
    if range_fre <= num_freq or num_freq <= 0:
        num_freq = range_fre
    freqs30 = freqs[:num_freq]
    mod_Dnu_mas30 = mod_Dnu[:num_freq] + Dnu
    mod_Dnu_stack30 = np.hstack(
        (mod_Dnu[:num_freq], mod_Dnu_mas30[mod_Dnu_mas30 < (Dnu + 2 * tol)])
    )  # We create another vector to account for periodicities close to 0 or 1
    fre_mas30 = np.hstack(
        (freqs30, freqs30[mod_Dnu_mas30 < (Dnu + 2 * tol)])
    )  # We need a new vector of the frequencies
    if range_fre > num_freq and range_fre <= 2 * num_freq:
        freqs60 = freqs[num_freq:]
        mod_Dnu_mas60 = mod_Dnu[num_freq:] + Dnu
        mod_Dnu_stack60 = np.hstack(
            (mod_Dnu[num_freq:], mod_Dnu_mas60[mod_Dnu_mas60 < (Dnu + 2 * tol)])
        )  # We create another vector to account for periodicities close to 0 or 1
        fre_mas60 = np.hstack(
            (freqs60, freqs60[mod_Dnu_mas60 < (Dnu + 2 * tol)])
        )  # We need a new vector of the frequencies
    elif range_fre > 2 * num_freq:
        freqs60 = freqs[num_freq : 2 * num_freq]
        mod_Dnu_mas60 = mod_Dnu[num_freq : 2 * num_freq] + Dnu
        mod_Dnu_stack60 = np.hstack(
            (
                mod_Dnu[num_freq : 2 * num_freq],
                mod_Dnu_mas60[mod_Dnu_mas60 < (Dnu + 2 * tol)],
            )
        )  # We create another vector to account for periodicities close to 0 or 1
        fre_mas60 = np.hstack(
            (freqs60, freqs60[mod_Dnu_mas60 < (Dnu + 2 * tol)])
        )  # We need a new vector of the frequencies

    # Now, we search for the frequencies separated by Dnu
    # Create a new vector with the frequencies in the diagram (we need this because we have "added" some frequencies in the previous step)
    if range_fre > num_freq:
        new_freqs = np.hstack((fre_mas30, fre_mas60))
        new_modDnu = np.hstack((mod_Dnu_stack30, mod_Dnu_stack60))
    else:
        new_freqs = fre_mas30
        new_modDnu = mod_Dnu_stack30

    # Matrix with all differences
    freq_matrix = abs(new_freqs[:, None] - new_freqs)
    il = np.tril_indices(np.size(new_freqs))  # Indices of the low matrix
    freq_matrix[il] = 0  # We only want the upper part of the matrix
    freq_matrix2 = np.matrix((freq_matrix))  # This is to find the 2Dnu differences
    freq_matrix[
        np.logical_or(freq_matrix < (Dnu - tol), freq_matrix > (Dnu + tol))
    ] = 0  # We also discard the values not separated by Dnu
    freq_matrix2[
        np.logical_or(freq_matrix2 < 2 * (Dnu - tol), freq_matrix2 > 2 * (Dnu + tol))
    ] = 0  # This matrix contains the values separated by 2*Dnu

    #    freq_matrix_test = freq_matrix[freq_matrix != 0]
    #    freq_matrix_test2 = freq_matrix2[freq_matrix2 != 0]

    index_Dnu_diff = np.where(freq_matrix != 0)  # Indices where differences = Dnu
    index_2Dnu_diff = np.where(freq_matrix2 != 0)  # Indices where differences = 2Dnu

    #    #Now, we order the vectors from low to high frequency
    #    n_ind = np.argsort(new_freqs)
    #    new_freqs = new_freqs[n_ind]
    #    new_modDnu = new_modDnu[n_ind]
    #
    #    #Now, the differences
    #    freqs_diff = np.diff(new_freqs) #Differences of consecutive frequencies
    #    n_diff = np.logical_and(freqs_diff > (Dnu - tol), freqs_diff < (Dnu + tol)) #Differences in the range [Dnu-tol, Dnu+tol]
    #    freqs_diff_Dnu = freqs_diff[n_diff]
    #    modDnu_diff_Dnu = new_modDnu[n_diff]
    #
    #    print new_freqs
    #    print freqs_diff
    #    print freqs_diff_Dnu,modDnu_diff_Dnu
    #
    #    freqs_diff2 = np.hstack((np.diff(new_freqs[::2]),np.diff(new_freqs[1::2]))) #Differences of frequencies separated by 2 in vector
    #    modDnu_diff_temp = np.hstack((new_modDnu[::2],new_modDnu[1::2]))
    #    n_diff2 = np.logical_and(freqs_diff2 > (Dnu - 2*tol), freqs_diff2 < (Dnu + 2*tol))
    #    freqs_diff_Dnu2 = freqs_diff2[n_diff2]
    #    modDnu_diff_Dnu2 = modDnu_diff_temp[n_diff2]
    #
    #    print freqs_diff_Dnu2,modDnu_diff_Dnu2

    # For the records: [t - s for s, t in zip(a, a[1:])]

    # Here PLOT the ECHELLE DIAGRAM
    fig_eche = plt.figure()
    ax_down_eche = fig_eche.add_subplot(1, 1, 1)
    (freq30_handles,) = ax_down_eche.plot(
        mod_Dnu_stack30 / Dnu, fre_mas30, "o", label=r"30 highest freqs"
    )
    if range_fre > num_freq:
        (freq60_handles,) = ax_down_eche.plot(
            mod_Dnu_stack60 / Dnu,
            fre_mas60,
            "o",
            markeredgecolor="grey",
            label=r"second 30 highest freqs",
        )
    # We also plot the connected frequencies
    connectedDnu = []
    for l in range(0, np.size(index_Dnu_diff[0])):
        x = (
            np.array(
                [new_modDnu[index_Dnu_diff[0][l]], new_modDnu[index_Dnu_diff[1][l]]]
            )
            / Dnu
        )
        y = np.array([new_freqs[index_Dnu_diff[0][l]], new_freqs[index_Dnu_diff[1][l]]])
        # We will not plot the connection between too far frequencies (just for visualization issues)
        if abs(x[1] - x[0]) <= 2 * tol / Dnu:
            # print x,y
            ax_down_eche.plot(x, y, "k")
        connectedDnu = np.hstack((connectedDnu, y))
    for l in range(0, np.size(index_2Dnu_diff[0])):
        x = (
            np.array(
                [new_modDnu[index_2Dnu_diff[0][l]], new_modDnu[index_2Dnu_diff[1][l]]]
            )
            / Dnu
        )
        y = np.array(
            [new_freqs[index_2Dnu_diff[0][l]], new_freqs[index_2Dnu_diff[1][l]]]
        )
        # We will not plot the connection between too far frequencies (just for visualization issues)
        if abs(x[1] - x[0]) <= 2 * tol / Dnu:
            # print x,y
            ax_down_eche.plot(x, y, "--", color="grey")
        connectedDnu = np.hstack((connectedDnu, y))

    connectedDnu = np.unique(connectedDnu)

    yminmax = ax_down_eche.get_ylim()
    xminmax = ax_down_eche.get_xlim()
    if xminmax[0] > 0:
        xminmax = (0, xminmax[1])  # To show the whole échelle diagram from 0 to 1
    if xminmax[1] < 1:
        xminmax = (xminmax[0], 1)  # To show the whole échelle diagram from 0 to 1
    ax_down_eche.axvline(Dnu / Dnu, color="k", linestyle="--")
    ax_down_eche.set_xlabel(
        r"Frequencies mod $\Delta\nu$ ("
        + "%.2f" % Dnu
        + " $\mu$Hz = "
        + "%.2f" % Dnu_d
        + " d$^{-1}$)"
    )
    ax_down_eche.set_ylabel("Frequencies ($\mu$Hz)")
    if range_fre > num_freq:
        ax_down_eche.legend(handles=[freq30_handles, freq60_handles], loc=2)
    else:
        ax_down_eche.legend(handles=[freq30_handles], loc=2)
    ax_up_eche = ax_down_eche.twinx()
    ax_up_eche.set_ylim([yminmax[0] / Dnu - 2, yminmax[1] / Dnu - 2])
    ax_up_eche.set_ylabel(r"Frequencies (units of $\sim$ n)")
    ax_up_eche.set_xlim([xminmax[0], xminmax[1]])
    if saveplot == "Y":
        if type(frec_f) is np.ndarray:
            fig_eche.savefig(
                "NotKnown_eche"
                + str(np.max(range_fre))
                + "freqs_Dnu"
                + "%.2f" % Dnu
                + ".eps",
                bbox_inches="tight",
            )
        elif os.path.isfile(frec_f):
            fig_eche.savefig(
                frec_f
                + "_eche"
                + str(np.max(range_fre))
                + "freqs_Dnu"
                + "%.2f" % Dnu
                + ".eps",
                bbox_inches="tight",
            )
    fig_eche.show()

    return mod_Dnu_stack30, fre_mas30, connectedDnu, fig_eche
