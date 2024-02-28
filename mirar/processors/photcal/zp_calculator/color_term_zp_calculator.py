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
            # unlikely galaxies if below sm_cutoff (near zero), likely galaxies if above (large positive value)
            ind_pos = np.where(matched_img_cat['SPREAD_MODEL'] > sm_cutoff)[0]
            ind_nearzero = np.where(matched_img_cat['SPREAD_MODEL'] < sm_cutoff)[0]

            sm_bad_tabs = matched_img_cat[ind_pos], matched_ref_cat[ind_pos]
            sm_good_tabs = matched_img_cat[ind_nearzero], matched_ref_cat[ind_nearzero]
            print('number of sources passing spread_model cut: ', len(sm_good_tabs[0]))

            # (psf - kron), indexes by tab
            kron_cutoff = 0.995 #0.990 #1
            white_psfmag = white_feat(matched_ref_cat, 'mag')
            white_kronmag = white_feat(matched_ref_cat, 'Kmag')
            white_psfkron_ratio = white_psfmag / white_kronmag
            likely_star_ind = np.where(white_psfkron_ratio > kron_cutoff)[0]
            likely_gal_ind = np.where(white_psfkron_ratio < kron_cutoff)[0]

            kron_bad_tabs = matched_img_cat[likely_gal_ind], matched_ref_cat[likely_gal_ind]
            kron_good_tabs = matched_img_cat[likely_star_ind], matched_ref_cat[likely_star_ind]
            print('number of sources passing kron cut: ', len(kron_good_tabs[0]))


            # define 3 sets of [x,y,x_err,y_err]. One for all, one for spread_model, one for kron_model
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

            y_notgal_sm = matched_ref_cat[ind_nearzero]["magnitude"] - matched_img_cat[ind_nearzero][colname]
            x_notgal_sm = matched_ref_cat[ind_nearzero][color_colnames[0]] - matched_ref_cat[ind_nearzero][
                color_colnames[1]]

            y_err_notgal_sm = np.sqrt(
                matched_img_cat[ind_nearzero][colname.replace("MAG", "MAGERR")] ** 2
                + matched_ref_cat[ind_nearzero]["magnitude_err"] ** 2
            )
            x_err_notgal_sm = np.sqrt(
                matched_ref_cat[ind_nearzero][color_err_colnames[0]] ** 2
                + matched_ref_cat[ind_nearzero][color_err_colnames[1]] ** 2
            )

            y_notgal_kron = matched_ref_cat[likely_star_ind]["magnitude"] - matched_img_cat[likely_star_ind][colname]
            x_notgal_kron = matched_ref_cat[likely_star_ind][color_colnames[0]] - matched_ref_cat[likely_star_ind][
                color_colnames[1]]
            y_err_notgal_kron = np.sqrt(
                matched_img_cat[likely_star_ind][colname.replace("MAG", "MAGERR")] ** 2
                + matched_ref_cat[likely_star_ind]["magnitude_err"] ** 2
            )
            x_err_notgal_kron = np.sqrt(
                matched_ref_cat[likely_star_ind][color_err_colnames[0]] ** 2
                + matched_ref_cat[likely_star_ind][color_err_colnames[1]] ** 2
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
            y_fit = line_func(out.beta, x)
            scatter_xy = np.std((out.beta[0] * x) - y)

            # use scipy.odr on spread_model now
            ## setup: remove sources with 0 uncertainty (or else scipy.odr won't work)
            zero_mask = (np.array(y_err_notgal_sm) == 0) | (np.array(x_err_notgal_sm) == 0)
            if np.sum(zero_mask) != 0:
                logger.debug(
                    f"Found {np.sum(zero_mask)} source(s) with zero reported "
                    f"uncertainty, removing them from calibrations."
                )
                x_notgal_sm, y_notgal_sm = x_notgal_sm[~zero_mask], y_notgal_sm[~zero_mask]
                x_err_notgal_sm, y_err_notgal_sm = x_err_notgal_sm[~zero_mask], y_err_notgal_sm[~zero_mask]
            line_model = Model(line_func)
            data_notgal_sm = RealData(x_notgal_sm, y_notgal_sm, sx=x_err_notgal_sm, sy=y_err_notgal_sm)
            odr_notgal_sm = ODR(data_notgal_sm, line_model, beta0=firstguess_color_zp)
            ## run the regression
            out_notgal_sm = odr_notgal_sm.run()
            color_notgal_sm, zero_point_notgal_sm = out_notgal_sm.beta
            color_err_notgal_sm, zp_err_notgal_sm = np.sqrt(np.diag(out_notgal_sm.cov_beta))
            y_fit_notgal_sm = line_func(out_notgal_sm.beta, x_notgal_sm)
            scatter_xy_notgal_sm = np.std((out_notgal_sm.beta[0] * x_notgal_sm) - y_notgal_sm)


            # use scipy.odr on kron now
            zero_mask = (np.array(y_err_notgal_kron) == 0) | (np.array(x_err_notgal_kron) == 0)
            if np.sum(zero_mask) != 0:
                logger.debug(
                    f"Found {np.sum(zero_mask)} source(s) with zero reported "
                    f"uncertainty, removing them from calibrations."
                )
                x_notgal_kron, y_notgal_kron = x_notgal_kron[~zero_mask], y_notgal_kron[~zero_mask]
                x_err_notgal_kron, y_err_notgal_kron = x_err_notgal_kron[~zero_mask], y_err_notgal_kron[~zero_mask]
            line_model = Model(line_func)
            data_notgal_kron = RealData(x_notgal_kron, y_notgal_kron, sx=x_err_notgal_kron, sy=y_err_notgal_kron)
            odr_notgal_kron = ODR(data_notgal_kron, line_model, beta0=firstguess_color_zp)
            ## run the regression
            out_notgal_kron = odr_notgal_kron.run()
            color_notgal_kron, zero_point_notgal_kron = out_notgal_kron.beta
            color_err_notgal_kron, zp_err_notgal_kron = np.sqrt(np.diag(out_notgal_kron.cov_beta))
            y_fit_notgal_kron = line_func(out_notgal_kron.beta, x_notgal_kron)
            scatter_xy_notgal_kron = np.std((out_notgal_kron.beta[0] * x_notgal_kron) - y_notgal_kron)


            # plot to visualize cuts + calibration!
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
            #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
            ax[0].set_xlabel(f'{color_colnames[0]} - {color_colnames[1]} color')
            magname = color_err_colnames[0].split('_')[1]
            ax[0].set_ylabel(f'PS1 {magname} - SEDMv2 {colname}')
            fig.suptitle(f'using {colname}')
            ax[0].set_title('color & zp fit')
            ## fit with all sources
            ax[0].errorbar(x, y, yerr=y_err, xerr=x_err, fmt='ko', label='all sources')
            ax[0].plot(x, y_fit, ls='-', label=f'ODR fit all c={out.beta[0]:.3f}$\pm${color_err:.3f}, '
                                            f'zp={out.beta[1]:.3f}$\pm${zp_err:.3f}', c='k', lw=3, alpha=0.5)
            ax[0].plot(x[0], y_fit[0], alpha=0, label=f'$\sigma$_y={scatter_xy:.5f}')

            ## fit with spread_model cuts
            ax[0].plot(x_notgal_sm, y_notgal_sm,
                    '*', c='orchid', ms=20, alpha=0.4, label=f'passed SM cut')#, sm_cutoff={sm_cutoff}')
            ax[0].plot(x_notgal_sm, (color_notgal_sm * x_notgal_sm) + zero_point_notgal_sm,
                    label=f'SM fit: c={color_notgal_sm:.3f}$\pm${color_err_notgal_sm:.3f}, '
                                                       f'zp={zero_point_notgal_sm:.3f}$\pm${zp_err_notgal_sm:.3f}', c='orchid')

            ## fit with kron cuts
            ax[0].plot(x_notgal_kron, y_notgal_kron,
                    '^', c='b', ms=15, alpha=0.25, label=f'passed kron cut')#, kron_cutoff={kron_cutoff}')
            ax[0].plot(x_notgal_kron, (color_notgal_kron * x_notgal_kron) + zero_point_notgal_kron,
                    label=f'Kron fit: c={color_notgal_kron:.3f}$\pm${color_err_notgal_kron:.3f}, '
                          f'zp={zero_point_notgal_kron:.3f}$\pm${zp_err_notgal_kron:.3f}', c='b')

            ax[0].legend()

            ax[1].set_title('SPREAD_MODEL cut')
            ax[1].hist(matched_img_cat['SPREAD_MODEL'], bins=11, color='k', alpha=0.9, label='sources')
            ax[1].set_xlabel('SPREAD_MODEL')
            ax[1].set_ylabel('counts')
            ax[1].axvline(sm_cutoff, label=f'cutoff={sm_cutoff}', color='orchid')
            ax[1].axvspan(ax[1].get_xlim()[0], sm_cutoff, alpha=0.4, color='orchid', label='included sources')
            ax[1].legend()

            ax[2].plot(white_kronmag, white_psfkron_ratio, 'o', ms=5, alpha=0.25, c='k', label='sources')
            ax[2].axhline(kron_cutoff, label=f'cutoff={kron_cutoff}', color='b')
            ax[2].set_title('Kron cut')
            ax[2].set_xlabel('whiteKronMag')
            ax[2].set_ylabel('whitePSFKronRatio')
            ax[2].axhspan(ax[2].get_ylim()[1], kron_cutoff, alpha=0.25, color='b', label='included sources')
            ax[2].legend()




            fig.savefig(f'/Users/saarahhall/Desktop/plots_nogal/{colname}_nogal_{sm_cutoff}_{kron_cutoff}.png')  # save the figure to file
            plt.close(fig)

            aperture = colname.split("_")[-1]
            image[f"ZP_{aperture}"] = zero_point
            image[f"ZP_{aperture}_std"] = zp_err
            image[f"ZP_{aperture}_nstars"] = len(matched_ref_cat)
            image[f"C_{aperture}"] = color
            image[f"C_{aperture}_std"] = color_err

        return image


def white_feat(ps1tab, feat_suffix):
    num_tosum = []
    denom_tosum = []
    for filt in ['g', 'r', 'i', 'z', 'y']:
        feat_col = filt + feat_suffix
        # a source is detected if PSFFlux_f, KronFlux_f, and ApFlux_f are all > 0
        mask_pos = np.where((ps1tab[filt + 'mag'] > 0) & (ps1tab[filt + 'Kmag'] > 0))  # and ApFlux.
        # TODO: we don't have Apflux in reference catalog
        # TODO: we still have masked values in the tab... but they don't appear in the returned array?
        # print(mask_pos)
        det = np.zeros(len(ps1tab))
        det[mask_pos] = 1

        # S/N squared in given filter
        w_f = (ps1tab[filt + 'Kmag'] / ps1tab['e_' + filt + 'Kmag']) ** 2
        feat = ps1tab[feat_col]  # kron mag? psf mag?

        num_tosum.append(w_f * feat * det)
        denom_tosum.append(w_f)

    nums = np.sum(num_tosum, axis=0)
    denoms = np.sum(denom_tosum, axis=0)

    return nums / denoms