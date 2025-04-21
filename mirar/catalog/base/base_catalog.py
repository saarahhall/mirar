"""
Module for Catalog base class
"""

import logging
from abc import ABC
from pathlib import Path
from typing import Type

import astropy.table

from mirar.catalog.base.errors import CatalogCacheError
from mirar.data import Image
from mirar.data.utils import get_image_center_wcs_coords
from mirar.paths import BASE_NAME_KEY, REF_CAT_PATH_KEY
from mirar.utils.ldac_tools import save_table_as_ldac

logger = logging.getLogger(__name__)

DEFAULT_SNR_THRESHOLD = 3.0


class ABCatalog:
    """
    Abstract class for catalog objects
    """

    @property
    def abbreviation(self):
        """
        Abbreviation for naming catalog files
        """
        raise NotImplementedError()

    def __init__(
        self,
        search_radius_arcmin: float,
    ):
        self.search_radius_arcmin = search_radius_arcmin


class BaseCatalog(ABCatalog, ABC):
    """
    Base class for catalog objects
    Attributes:
        min_mag: Minimum magnitude for stars in catalog
        max_mag: Maximum magnitude for stars in catalog
        filter_name: Filter name for catalog
        cache_catalog_locally: Whether to cache catalog locally?
        catalog_cachepath_key: Header key that stores the full path to the cached
        catalog. Recommended to use the inbuilt REF_CAT_PATH_KEY.
        Users need to add this to the image header themselves. e.g. For winter,
        we use a CustomImageModifer to add this key to the header, as our catalogs are
        cached by field-id, subdet-id and filter.
    """

    def __init__(
        self,
        *args,
        min_mag: float,
        max_mag: float,
        filter_name: str,
        cache_catalog_locally: bool = False,
        catalog_cachepath_key: str = REF_CAT_PATH_KEY,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.filter_name = filter_name
        self.cache_catalog_locally = cache_catalog_locally
        self.catalog_cachepath_key = catalog_cachepath_key

    def get_catalog(self, ra_deg: float, dec_deg: float) -> astropy.table.Table:
        """
        Returns a catalog centered on ra/dec

        :param ra_deg: RA
        :param dec_deg: Dec
        :return: Catalog
        """
        raise NotImplementedError()

    def write_catalog(self, image: Image, output_dir: str | Path) -> Path:
        """
        Generates a custom catalog for an image

        :param image: Image
        :param output_dir: output directory for catalog
        :return: path of catalog
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        ra_deg, dec_deg = get_image_center_wcs_coords(image, origin=1)

        base_name = Path(image[BASE_NAME_KEY]).with_suffix(".ldac").name

        cat = self.get_catalog(ra_deg=ra_deg, dec_deg=dec_deg)

        output_path = self.get_output_path(output_dir, base_name)
        output_path.unlink(missing_ok=True)

        logger.debug(f"Saving catalog to {output_path}")

        save_table_as_ldac(cat, output_path)

        if self.cache_catalog_locally:
            if self.catalog_cachepath_key not in image.header:
                err = (
                    f"Catalog caching requested, but "
                    f"{self.catalog_cachepath_key} not found in image header"
                )
                raise CatalogCacheError(err)
            catalog_save_path = Path(image[self.catalog_cachepath_key])
            catalog_save_path.unlink(missing_ok=True)
            logger.debug(f"Saving catalog to {catalog_save_path}")
            save_table_as_ldac(cat, catalog_save_path)

        return output_path

    def get_output_path(self, output_dir: Path, base_name: str | Path) -> Path:
        """
        Get save path for catalog

        :param output_dir: Output directory for catalog
        :param base_name: Base name for catalog
        :return: Full output path
        """
        cat_base_name = Path(base_name).with_suffix(f".{self.abbreviation}.cat")
        return output_dir.joinpath(cat_base_name)


class BaseMultiBackendCatalog(ABC):
    """
    Base class for composite catalogs that are made up of multiple catalog backends
    """

    def __new__(cls, *args, backend: str | None = None, **kwargs):
        backend_class = cls.set_backend(backend)
        return backend_class(*args, **kwargs)

    @staticmethod
    def set_backend(backend: str | None) -> Type[BaseCatalog]:
        """
        Set backend for composite catalog

        :param backend: Backend name
        """
        raise NotImplementedError()
