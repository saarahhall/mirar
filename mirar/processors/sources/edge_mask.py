"""
Edge mask processor
"""

import logging

import numpy as np

from mirar.data import SourceBatch
from mirar.processors.base_processor import BaseSourceProcessor

logger = logging.getLogger(__name__)


class EdgeSourcesMask(BaseSourceProcessor):
    """
    Class to mask sources near the edge of the image
    """

    base_key = "egdemask"

    def __init__(
        self,
        edge_boundary_size: float,
        image_xsize_column_key: str = "X_SHAPE",
        image_ysize_column_key: str = "Y_SHAPE",
        x_column_key: str = "X_IMAGE",
        y_column_key: str = "Y_IMAGE",
    ):
        super().__init__()

        self.edge_boundary_size = edge_boundary_size
        self.image_xsize_column_key = image_xsize_column_key
        self.image_ysize_column_key = image_ysize_column_key
        self.x_column_key = x_column_key
        self.y_column_key = y_column_key

    def description(self) -> str:
        return f"Mask sources within {self.edge_boundary_size} pixels of image edge"

    def _apply_to_sources(
        self,
        batch: SourceBatch,
    ) -> SourceBatch:
        for source_table in batch:
            candidate_table = source_table.get_data()
            x_coords = candidate_table[self.x_column_key]
            y_coords = candidate_table[self.y_column_key]
            image_xsize = candidate_table[self.image_xsize_column_key]
            image_ysize = candidate_table[self.image_ysize_column_key]
            logger.debug(f"Applying edge-filter to {len(candidate_table)} candidates")
            near_x_edge_mask = (x_coords < self.edge_boundary_size) | (
                x_coords > image_xsize - self.edge_boundary_size
            )
            near_y_edge_mask = (y_coords < self.edge_boundary_size) | (
                y_coords > image_ysize - self.edge_boundary_size
            )
            near_edge_mask = near_y_edge_mask | near_x_edge_mask
            masked_table = candidate_table[np.invert(near_edge_mask)]
            logger.debug(f"Edge-filter passed {len(masked_table)} candidates")
            source_table.set_data(candidate_table)

        return batch
