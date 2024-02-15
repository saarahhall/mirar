"""
Module to calculate zero point by including a color term.
"""

import logging
from typing import Callable

import numpy as np
from astropy.table import Table
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt

from mirar.data import Image
from mirar.processors.photcal.zp_calculator.base_zp_calculator import (
    BaseZeroPointCalculator,
)

logger = logging.getLogger(__name__)


def line_func(theta, x):
    """
    linear model to hand to scipy.odr
    Args:
        theta: slope and intercept of line to fit to data
        x: x data (color in our context)

    Returns:

    """
    slope, intercept = theta
    return (slope * x) + intercept


class ZPWithColorTermCalculator(
    BaseZeroPointCalculator
):  # pylint: disable=too-few-public-methods
    """
    Class to calculate zero point by including a color term. This models the data as

    ref_mag - img_mag = ZP + C * (ref_color)

    Attributes:
        color_colnames_generator: function that takes an image as input and returns
        two lists containing two strings each that are the column names of the reference
        catalog magnitudes and magnitude errors to use for the color term. The first
        string is the bluer band, the second is the redder band.
    """

    def __init__(
        self,
        color_colnames_generator: Callable[
            [Image], list[list[str, str], list[str, str]]
        ],
    ):
        self.color_colnames_generator = color_colnames_generator

    def calculate_zeropoint(  # pylint: disable=too-many-locals
        self,
        image: Image,
        matched_ref_cat: Table,
        matched_img_cat: Table,
        colnames: list[str],
    ) -> Image:
        (
            color_colnames,
            color_err_colnames,
            firstguess_color_zp,
        ) = self.color_colnames_generator(image)
        colors = matched_ref_cat[color_colnames[0]] - matched_ref_cat[color_colnames[1]]

        for colname in colnames:

            # (spread_model), indexes by tab
            sm_cutoff = 0.0015
            ind_pos = np.where(matched_img_cat['SPREAD_MODEL'] > sm_cutoff)[0]
            bad_tabs = matched_img_cat[ind_pos], matched_ref_cat[ind_pos]
            # points that are unlikely galaxies
            ind_nearzero = np.where(matched_img_cat['SPREAD_MODEL'] < cutoff)[0]
            y_notgal_sm = matched_ref_cat[ind_nearzero]["magnitude"] - matched_img_cat[ind_nearzero][colname]
            x_notgal_sm = matched_ref_cat[ind_nearzero][color_colnames[0]] - matched_ref_cat[ind_nearzero][color_colnames[1]]
            y_err_notgal_sm = np.sqrt(
                matched_img_cat[ind_nearzero][colname.replace("MAG", "MAGERR")] ** 2
                + matched_ref_cat[ind_nearzero]["magnitude_err"] ** 2
            )
            x_err_notgal_sm = np.sqrt(
                matched_ref_cat[ind_nearzero][color_err_colnames[0]] ** 2
                + matched_ref_cat[ind_nearzero][color_err_colnames[1]] ** 2
            )

            # (psf - kron), indexes by tab



            y = matched_ref_cat["magnitude"] - matched_img_cat[colname]
            x = colors
            y_err = np.sqrt(
                matched_img_cat[colname.replace("MAG", "MAGERR")] ** 2
                + matched_ref_cat["magnitude_err"] ** 2
            )
            x_err = np.sqrt(
                matched_ref_cat[color_err_colnames[0]] ** 2
                + matched_ref_cat[color_err_colnames[1]] ** 2
            )


            # use scipy.odr to fit a line to data with x and y uncertainties
            ## setup: remove sources with 0 uncertainty (or else scipy.odr won't work)
            zero_mask = (np.array(y_err) == 0) | (np.array(x_err) == 0)
            if np.sum(zero_mask) != 0:
                logger.debug(
                    f"Found {np.sum(zero_mask)} source(s) with zero reported "
                    f"uncertainty, removing them from calibrations."
                )
                x, y = x[~zero_mask], y[~zero_mask]
                x_err, y_err = x_err[~zero_mask], y_err[~zero_mask]

            ## set up odr
            line_model = Model(line_func)
            data = RealData(x, y, sx=x_err, sy=y_err)
            odr = ODR(data, line_model, beta0=firstguess_color_zp)
            ## run the regression
            out = odr.run()
            color, zero_point = out.beta
            color_err, zp_err = np.sqrt(np.diag(out.cov_beta))

            aperture = colname.split("_")[-1]
            image[f"ZP_{aperture}"] = zero_point
            image[f"ZP_{aperture}_std"] = zp_err
            image[f"ZP_{aperture}_nstars"] = len(matched_ref_cat)
            image[f"C_{aperture}"] = color
            image[f"C_{aperture}_std"] = color_err

            y_fit = line_func(out.beta, x)
            # plot invisible point so label (scatter statistic) appears in legend
            scatter_xy = np.std((out.beta[0] * x) - y)

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.errorbar(x, y, yerr=y_err, xerr=x_err, fmt='ko', label='all sources')
            ax.set_xlabel(f'{color_colnames[0]} - {color_colnames[1]} color')
            magname = color_err_colnames[0].split('_')[1]
            ax.set_ylabel(f'PS1 {magname} - SEDMv2 {colname}')
            ax.set_title(f'using {colname}')

            ax.plot(x, y_fit, ls='-', label=f'ODR fit c={out.beta[0]:.3f}$\pm${color_err:.3f}, '
                                            f'zp={out.beta[1]:.3f}$\pm${zp_err:.3f}', c='hotpink', lw=3, alpha=0.5)
            ax.plot(x[0], y_fit[0], alpha=0, label=f'$\sigma$_y={scatter_xy:.5f}')

            # spread_model cuts
            # ax.plot(bad_tabs[1][color_colnames[0]] - bad_tabs[1][color_colnames[1]],
            #        bad_tabs[1]["magnitude"] - bad_tabs[0][colname],
            #        '*', c='orchid', ms=20, alpha=0.4, label=f'spread_model>{cutoff}')
            # ax.plot(x_notgal, (color_notgal * x_notgal) + zero_point_notgal,
            #        label=f'fit notgal: c={color_notgal:.3f}$\pm${color_err_notgal:.3f}, '
            #                                           f'zp={zero_point_notgal:.3f}$\pm${zp_err_notgal:.3f}', c='orchid')
            ax.legend()


            fig.savefig(f'/Users/saarahhall/Desktop/{colname}.png')  # save the figure to file
            plt.close(fig)

        return image
