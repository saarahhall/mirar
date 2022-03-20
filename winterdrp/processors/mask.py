import astropy.io.fits
import numpy as np
import logging
from winterdrp.processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class MaskPixels(BaseProcessor):

    base_key = "mask"

    def __init__(
            self,
            mask_path: str,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask = None
        self.mask_path = mask_path

    def get_mask(self):
        if self.mask is None:
            self.mask, _ = self.open_fits(self.mask_path)
        return self.mask

    def _apply_to_images(
            self,
            images: list[np.ndarray],
            headers: list[astropy.io.fits.Header],
    ) -> tuple[list[np.ndarray], list[astropy.io.fits.Header]]:

        for i, data in enumerate(images):
            header = headers[i]
            mask = self.get_mask()

            mask = mask != 0

            data[mask] = np.nan
            images[i] = data
            headers[i] = header

        return images, headers