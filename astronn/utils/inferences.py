import logging

from astronn.utils.metrics import *

import numpy as np
from scipy.signal import find_peaks, peak_widths

import matplotlib.pyplot as plt
import pandas as pd


class inferences:
    """
    Utils to calculate and preprocess inferences from NN
    """

    def get_peak_width(self, probs, position, peaks, peaks_width, peaks_sorted_by_prob):
        """
        """
        peak_id = np.where(probs == peaks_sorted_by_prob[position])[0][0]
        return (peak_id, peaks_width[0][np.where(peaks == peak_id)[0][0]])

    def get_processed_inferences(
        self, nn_model, df_stars, take_number, csv_L=False, debug=False
    ):
        """
        """
        # Save results and predictions
        results = {}

        for star in df_stars.take(take_number):  # take the take_number first stars
            target = np.where(star[2].numpy().flat == 1)[0].flat[0]

            # Save results
            if star[0].numpy()[0].decode("utf-8") not in results:
                results[star[0].numpy()[0].decode("utf-8")] = {
                    "dnu-target": target,
                }

            # Get probabilities given by NN
            probs = nn_model.predict(star[1])[0]
            # Get peaks
            peaks, _ = find_peaks(probs, height=0, distance=10)
            peaks_width = peak_widths(probs, peaks)
            # Sort peajs by probability
            peaks_sorted_by_prob = np.sort(probs[peaks])[::-1]

            # save top-k=1
            best_peak, best_peak_width = self.get_peak_width(
                probs, 0, peaks, peaks_width, peaks_sorted_by_prob
            )
            results[star[0].numpy()[0].decode("utf-8")]["top1"] = best_peak
            # Get error
            results[star[0].numpy()[0].decode("utf-8")]["e-top1"] = best_peak_width

            # save top-k=2
            best_peak, best_peak_width = self.get_peak_width(
                probs, 1, peaks, peaks_width, peaks_sorted_by_prob
            )
            results[star[0].numpy()[0].decode("utf-8")]["top2"] = best_peak
            # Get error
            results[star[0].numpy()[0].decode("utf-8")]["e-top2"] = best_peak_width

            # save top-k=3
            best_peak, best_peak_width = self.get_peak_width(
                probs, 2, peaks, peaks_width, peaks_sorted_by_prob
            )
            results[star[0].numpy()[0].decode("utf-8")]["top3"] = best_peak
            # Get error
            results[star[0].numpy()[0].decode("utf-8")]["e-top3"] = best_peak_width

            # save top-k=4
            best_peak, best_peak_width = self.get_peak_width(
                probs, 3, peaks, peaks_width, peaks_sorted_by_prob
            )
            results[star[0].numpy()[0].decode("utf-8")]["top4"] = best_peak
            # Get error
            results[star[0].numpy()[0].decode("utf-8")]["e-top4"] = best_peak_width

            if csv_L is not None:
                # Get P from L
                # Check for non L in csv or NaN value
                L = csv_L[csv_L.ID.eq(star[0].numpy()[0].decode("utf-8").upper())][
                    "L"
                ].values
                if len(L) > 0 and not np.isnan(L):
                    P = get_P(np.log10(L))
                    p_e = p_error(log_L=np.log10(L))
                    rho_q_up = get_rho_from_P(P - p_e, Q=0.042)
                    rho_q_down = get_rho_from_P(P + p_e, Q=0.033)
                    results[star[0].numpy()[0].decode("utf-8")]["dnu-from-P-up"] = (
                        get_dnu_from_rho(rho_q_up[0]) * dnu_sun
                    )
                    results[star[0].numpy()[0].decode("utf-8")]["dnu-from-P-down"] = (
                        get_dnu_from_rho(rho_q_down[0]) * dnu_sun
                    )
                else:
                    # print(star[0].numpy()[0].decode("utf-8"))
                    # print(L)
                    results[star[0].numpy()[0].decode("utf-8")][
                        "dnu-from-P-up"
                    ] = np.nan
                    results[star[0].numpy()[0].decode("utf-8")][
                        "dnu-from-P-down"
                    ] = np.nan

            # Plot star info
            if debug:
                x = np.arange(0, 100, 0.25)  # x axis from 0 to 100
                plt.plot(x, star[1][0, :, 0], label="dft", color="blue")
                # plt.plot(x, star[1][0, :, 1], label="hod", color="orange", alpha=0.5)
                plt.plot(x, star[1][0, :, 1], label="ac", color="green")
                plt.axvline(
                    x=results[star[0].numpy()[0].decode("utf-8")]["top1"],
                    label="top1",
                    linestyle="dashed",
                    linewidth=3,
                )
                plt.axvline(
                    x=results[star[0].numpy()[0].decode("utf-8")]["top2"],
                    label="top2",
                    linestyle="dashed",
                    linewidth=2,
                )
                plt.axvline(
                    x=results[star[0].numpy()[0].decode("utf-8")]["top3"],
                    label="top3",
                    linestyle="dashed",
                    linewidth=1,
                )
                plt.axvline(
                    x=results[star[0].numpy()[0].decode("utf-8")]["top4"],
                    label="top4",
                    linestyle="dashed",
                    linewidth=1,
                )
                plt.axvline(x=target, label="target", color="red")
                plt.title(star[0].numpy()[0].decode("utf-8"))
                plt.legend(title="Channels")
                plt.show()
        df = pd.DataFrame(
            columns=[
                "id",
                "dnu-target",
                "top1",
                "e-top1",
                "top2",
                "e-top2",
                "top3",
                "e-top3",
                "top4",
                "e-top4",
                "dnu-from-P-up",
                "dnu-from-P-down",
            ]
        )
        for i, id in enumerate(results):
            df.loc[i] = [
                id,
                results[id]["dnu-target"],
                results[id]["top1"],
                results[id]["e-top1"],
                results[id]["top2"],
                results[id]["e-top2"],
                results[id]["top3"],
                results[id]["e-top3"],
                results[id]["top4"],
                results[id]["e-top4"],
                results[id]["dnu-from-P-up"],
                results[id]["dnu-from-P-down"],
            ]
        # Derive rho from Dnu
        df["rho-target"] = get_rho(df["dnu-target"] / dnu_sun)
        df["e-rho-target"] = rho_error(np.float64(df["dnu-target"].values / dnu_sun))
        df["rho-top1"] = get_rho(df["top1"] / dnu_sun)
        df["e-rho-top1"] = rho_error(np.float64(df["top1"].values / dnu_sun))
        df["rho-top2"] = get_rho(df["top2"] / dnu_sun)
        df["e-rho-top2"] = rho_error(np.float64(df["top2"].values / dnu_sun))
        df["rho-top3"] = get_rho(df["top3"] / dnu_sun)
        df["e-rho-top3"] = rho_error(np.float64(df["top3"].values / dnu_sun))
        df["rho-top4"] = get_rho(df["top4"] / dnu_sun)
        df["e-rho-top4"] = rho_error(np.float64(df["top4"].values / dnu_sun))

        return df

    def plot_inferences(self, df, plot_title="", plot_size=(7, 4)):
        """
        """
        plt.subplots(1, figsize=plot_size, dpi=120)

        plt.errorbar(
            df["id"],
            df["dnu-target"],
            0,
            fmt="x",
            markersize=10,
            capsize=2,
            label="Dnu",
        )

        plt.errorbar(
            df["id"],
            df["top1"],
            df["e-top1"],
            elinewidth=0.5,
            markersize=6,
            capsize=2,
            fmt="o",
            color="red",
            label="top-1",
        )
        plt.errorbar(
            df["id"],
            df["top2"],
            df["e-top2"],
            elinewidth=0.5,
            markersize=3,
            capsize=2,
            fmt="o",
            color="orange",
            label="top-2",
        )
        plt.errorbar(
            df["id"],
            df["top3"],
            df["e-top3"],
            elinewidth=0.5,
            markersize=2,
            capsize=2,
            fmt="o",
            color="green",
            label="top-3",
        )
        """
        plt.errorbar(
            df["id"],
            df["top4"],
            df["e-top4"],
            elinewidth=0.5,
            markersize=3,
            capsize=2,
            fmt="o",
            color="blue",
            label="top-4",
        )

        plt.errorbar(
            df["id"],
            (df["dnu-from-P-up"] + df["dnu-from-P-down"]) / 2,
            df["dnu-from-P-up"] - ((df["dnu-from-P-up"] + df["dnu-from-P-down"]) / 2),
            elinewidth=8,
            fmt="o",
            markersize=0,
            capsize=0,
            label="PL",
            color="lightblue",
        )
        """
        plt.xticks(rotation=90)

        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            title=r"$\Delta\nu$ from NN",
        )
        # plt.gca().add_artist(legend1)
        plt.ylabel("$\langle\Delta\\nu\\rangle / \Delta\\nu_\odot$")
        plt.title(plot_title)
        # plt.ylim(0, 1.0)
        plt.show()

    def plot_relation_rodriguez(df, labels=True, relation_line_range=(3.2, 9)):
        """
        """
        plt.subplots(1, figsize=(7, 4), dpi=120)

        select_closest_top = np.argmin(
            (
                np.power(np.asarray(df["dnu-target"]) - np.asarray(df["top1"]), 2,),
                np.power(np.asarray(df["dnu-target"]) - np.asarray(df["top2"]), 2,),
            ),
            axis=0,
        )

        tops = np.where(select_closest_top == 0, df["top1"], df["top2"],).astype(float)

        if labels:
            for i, row in df.iterrows():
                plt.annotate(
                    row["id"],
                    (
                        np.log10(tops[i] / dnu_sun),
                        np.log10(
                            get_rho(np.asarray(row["dnu-target"]) / dnu_sun) / rho_sun
                        ),
                    ),
                    size=9,
                )

        plt.scatter(
            np.log10(tops / dnu_sun),
            np.log10(
                get_rho(np.asarray(df["dnu-target"].values.astype(float)) / dnu_sun)
                / rho_sun
            ),
        )

        # Plot relation
        xs = np.arange(relation_line_range[0], relation_line_range[1])
        rs = get_rho(xs)
        plt.plot(
            np.log10(rs / dnu_sun),
            np.log10(get_rho(np.asarray(rs) / dnu_sun) / rho_sun),
            label="Relation (M)",
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.title("Closest NN top on the Rodriguez-Martin et.al. 2020 relation")
        plt.ylabel("$\log(\\rho/\\rho_\odot)$")
        plt.xlabel("$\log(\Delta\\nu/\Delta\\nu_\odot)$")

        plt.show()
