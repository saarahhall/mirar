"""
Script containing the various
:class:`~mirar.processors.base_processor.BaseProcessor`
lists which are used to build configurations for the
:class:`~mirar.pipelines.sedmv2.sedmv2_pipeline.SEDMv2Pipeline`.
"""

from mirar.paths import BASE_NAME_KEY, OBSCLASS_KEY
from mirar.pipelines.sedmv2.config import (
    psfex_config_path,
    sedmv2_mask_path,
    sextractor_astrometry_config,
    sextractor_photometry_config,
    sextractor_PSF_photometry_config,
    swarp_config_path,
)
from mirar.pipelines.sedmv2.config.constants import SEDMV2_PIXEL_SCALE
from mirar.pipelines.sedmv2.generator import (
    sedmv2_color_function_ps1,
    sedmv2_photcal_catalog_purifier,
    sedmv2_photometric_catalog_generator,
    sedmv2_reference_image_generator,
    sedmv2_reference_image_resampler,
    sedmv2_reference_psfex,
    sedmv2_reference_sextractor,
    sedmv2_zogy_catalogs_purifier,
)
from mirar.pipelines.sedmv2.load_sedmv2_image import load_sedmv2_mef_image
from mirar.processors import BiasCalibrator, FlatCalibrator
from mirar.processors.astromatic import PSFex, Sextractor, Swarp
from mirar.processors.astromatic.sextractor.background_subtractor import (
    SextractorBkgSubtractor,
)
from mirar.processors.astrometry.anet import AstrometryNet
from mirar.processors.dark import DarkCalibrator
from mirar.processors.mask import MaskPixelsFromPath
from mirar.processors.photcal.photcalibrator import PhotCalibrator
from mirar.processors.photcal.zp_calculator import (
    OutlierRejectionZPCalculator,
    ZPWithColorTermCalculator,
)
from mirar.processors.photometry import AperturePhotometry, PSFPhotometry
from mirar.processors.reference import ProcessReference
from mirar.processors.skyportal.skyportal_source import SkyportalSourceUploader
from mirar.processors.sources import (
    ForcedPhotometryDetector,
    SextractorSourceDetector,
    SourceWriter,
)
from mirar.processors.utils import (
    ImageBatcher,
    ImageDebatcher,
    ImageSaver,
    ImageSelector,
    MEFLoader,
)
from mirar.processors.utils.header_annotate import HeaderEditor
from mirar.processors.zogy.zogy import ZOGY, ZOGYPrepare

load_raw = [
    MEFLoader(
        input_sub_dir="",
        load_image=load_sedmv2_mef_image,
    ),
]

mask_all = [  # masks cals and science, assuming run just after load_raw
    MaskPixelsFromPath(mask_path=sedmv2_mask_path)
]

bias_all = [  # applies bias subtraction to darks, flats, and science
    ImageDebatcher(),
    BiasCalibrator(),
]

dark_all = [  # applies dark subtraction to flats and science
    ImageDebatcher(),
    ImageSelector((OBSCLASS_KEY, ["dark", "flat", "science"])),
    DarkCalibrator(cache_sub_dir="calibration_darks"),
]

flat_all = [
    ImageSelector((OBSCLASS_KEY, ["flat", "science"])),
    ImageBatcher(split_key="filterid"),
    FlatCalibrator(),
    ImageSaver(output_dir_name="detrend", write_mask=True),
]

skysub = [
    ImageBatcher(split_key=BASE_NAME_KEY),
    ImageSelector((OBSCLASS_KEY, ["science"])),  # pylint: disable=duplicate-code
    Sextractor(
        **sextractor_astrometry_config,
        write_regions_bool=True,
        cache=True,
        output_sub_dir="skysub",
        checkimage_type=["-BACKGROUND"],
    ),
    SextractorBkgSubtractor(),
]

reduce_not0 = mask_all + bias_all + dark_all + flat_all + skysub
reduce_0 = mask_all + dark_all + flat_all + skysub
reduce_no_dark = mask_all + bias_all + flat_all + skysub

astrometry = [
    AstrometryNet(
        output_sub_dir="a-net",
        scale_bounds=(0.08333333, 0.11666667),
        scale_units="degw",
        downsample=2,
        timeout=60,
        use_sextractor=True,
    ),
    Sextractor(
        output_sub_dir="sextractor",
        checkimage_name=None,
        checkimage_type=None,
        **sextractor_astrometry_config,
    ),
]

calibrate = [
    Sextractor(
        output_sub_dir="photprocess",
        checkimage_type="BACKGROUND_RMS",
        **sextractor_photometry_config,
    ),  # pylint: disable=duplicate-code
    PhotCalibrator(ref_catalog_generator=sedmv2_photometric_catalog_generator),
    ImageSaver(
        output_dir_name="processed",
        write_mask=True,
    ),
    HeaderEditor(edit_keys="procflag", values=1),
]


# stellar --

resample_stellar = [
    # ImageDebatcher(),
    # reaches for files coming from the same object
    # (note there can be more tha one MEF file per stellar object!)
    # ImageBatcher(split_key="OBJECTID"),
    Swarp(
        swarp_config_path=swarp_config_path,
        include_scamp=False,
        combine=False,
        calculate_dims_in_swarp=True,
    ),
    ImageSaver(
        output_dir_name="resampled", write_mask=True
    ),  # pylint: disable=duplicate-code
]


parse_stellar = [ImageSelector(("SOURCE", ["stellar", "None"]))]

process_stellar = reduce_not0 + astrometry + resample_stellar + calibrate

image_photometry = [  # imported from wirc/blocks.py
    ForcedPhotometryDetector(ra_header_key="OBJRAD", dec_header_key="OBJDECD"),
    AperturePhotometry(
        aper_diameters=[
            2 / SEDMV2_PIXEL_SCALE,
            3 / SEDMV2_PIXEL_SCALE,
            4 / SEDMV2_PIXEL_SCALE,
            5 / SEDMV2_PIXEL_SCALE,
            10 / SEDMV2_PIXEL_SCALE,
        ],
        bkg_in_diameters=[
            2.5 / SEDMV2_PIXEL_SCALE,
            3.5 / SEDMV2_PIXEL_SCALE,
            4.5 / SEDMV2_PIXEL_SCALE,
            5.5 / SEDMV2_PIXEL_SCALE,
            10.5 / SEDMV2_PIXEL_SCALE,
        ],
        bkg_out_diameters=[
            5.5 / SEDMV2_PIXEL_SCALE,
            8.6 / SEDMV2_PIXEL_SCALE,
            9.5 / SEDMV2_PIXEL_SCALE,
            10.6 / SEDMV2_PIXEL_SCALE,
            15.6 / SEDMV2_PIXEL_SCALE,
        ],
        col_suffix_list=["2", "3", "4", "5", "10"],
        phot_cutout_half_size=100,
        zp_key="ZP_AUTO",
    ),
    SourceWriter(output_dir_name="sourcetable"),
]


# transients --

parse_transient = [ImageSelector(("SOURCE", ["transient", "None"]))]

resample_transient = [
    ImageDebatcher(),
    ImageBatcher(split_key="origname"),  # reaches for files coming from the same MEF
    Swarp(
        # cache=True,
        swarp_config_path=swarp_config_path,
        include_scamp=False,
        combine=True,
        calculate_dims_in_swarp=True,
    ),
    ImageSaver(
        output_dir_name="resampled", write_mask=True
    ),  # pylint: disable=duplicate-code
]

transient_phot = [  # run phot on target in image
    PSFex(config_path=psfex_config_path, norm_fits=True),
    ForcedPhotometryDetector(ra_header_key="OBJRAD", dec_header_key="OBJDECD"),
    PSFPhotometry(),
    SourceWriter(output_dir_name="sourcetable"),
]

transient_phot_psfexsex = [  # run phot on target in image with new PSF method
    PSFex(config_path=psfex_config_path, norm_fits=True),
    Sextractor(
        output_sub_dir="photprocess",
        checkimage_type="BACKGROUND_RMS",
        use_psfex=True,
        **sextractor_PSF_photometry_config,
    ),  # Sextractor-based PSF mags, saves to catalog
    SextractorSourceDetector(output_sub_dir="sources", target_only=True),
    SourceWriter(output_dir_name="sourcetable"),
]

psf_all_sources_before_calibration = [  # run phot on all sources in image
    Sextractor(
        output_sub_dir="sextractor_before_psfex",
        checkimage_type="BACKGROUND_RMS",
        **sextractor_photometry_config,
    ),  # pylint: disable=duplicate-code
    PSFex(config_path=psfex_config_path, norm_fits=True),
    Sextractor(
        output_sub_dir="sextractor_after_psfex",
        checkimage_type="BACKGROUND_RMS",
        use_psfex=True,
        **sextractor_PSF_photometry_config,
    ),  # Sextractor-based PSF mags, saves to catalog
]

photcal_withcolor = [
    PhotCalibrator(
        ref_catalog_generator=sedmv2_photometric_catalog_generator,
        catalogs_purifier=sedmv2_photcal_catalog_purifier,
        zp_calculator=ZPWithColorTermCalculator(
            color_colnames_guess_generator=sedmv2_color_function_ps1,
            reject_outliers=True,
            solver="odr",
        ),
        write_regions=True,
    ),
    ImageSaver(
        output_dir_name="processed_after_psf",
        write_mask=True,
    ),
]

photcal_withoutcolor = [
    PhotCalibrator(
        ref_catalog_generator=sedmv2_photometric_catalog_generator,
        catalogs_purifier=sedmv2_photcal_catalog_purifier,
        zp_calculator=OutlierRejectionZPCalculator(),
        write_regions=True,
    ),
    ImageSaver(
        output_dir_name="processed_after_psf",
        write_mask=True,
    ),
]

all_phot_psfexsex_calibrate_nocolor = (
    psf_all_sources_before_calibration + photcal_withoutcolor
)
all_phot_psfexsex_calibrate = psf_all_sources_before_calibration + photcal_withcolor


process_all_psf_then_cal = (
    reduce_not0 + astrometry + resample_transient + all_phot_psfexsex_calibrate
)
process_all_psf_then_cal_no_color_term = (
    reduce_not0 + astrometry + resample_transient + all_phot_psfexsex_calibrate_nocolor
)

process_transient_using_magauto = (
    reduce_not0 + astrometry + resample_transient + calibrate
)

imsub = [
    ImageBatcher(split_key=BASE_NAME_KEY),
    ImageSelector((OBSCLASS_KEY, "science")),
    ProcessReference(
        ref_image_generator=sedmv2_reference_image_generator,
        ref_psfex=sedmv2_reference_psfex,
        sextractor=sedmv2_reference_sextractor,
        swarp_resampler=sedmv2_reference_image_resampler,  # pylint: disable=duplicate-code
    ),
    Sextractor(
        output_sub_dir="subtract",
        cache=False,
        write_regions_bool=True,
        **sextractor_photometry_config,
    ),
    PSFex(config_path=psfex_config_path, output_sub_dir="subtract", norm_fits=True),
    ImageSaver(output_dir_name="ref"),
    ZOGYPrepare(
        output_sub_dir="subtract",
        sci_zp_header_key="ZP_AUTO",
        ref_zp_header_key="ZPT_0000",
        catalog_purifier=sedmv2_zogy_catalogs_purifier,
    ),
    ZOGY(output_sub_dir="subtract"),
]


psf_phot_after_imsub = [
    ForcedPhotometryDetector(ra_header_key="OBJRAD", dec_header_key="OBJDECD"),
    PSFPhotometry(phot_cutout_half_size=10, zp_key="ZP_PSF", zp_std_key="ZP_PSF_std"),
    SourceWriter(output_dir_name="candidates"),
]


upload_fritz = [
    SkyportalSourceUploader(
        origin="SEDMv2TEST",
        group_ids=[1423],
        instrument_id=1078,
        update_thumbnails=False,
    )
]
