"""
Module with classes to write a regions file from a pandas dataframe
"""

import logging
import os
from pathlib import Path
from typing import Optional

from mirar.data import SourceBatch
from mirar.paths import (
    BASE_NAME_KEY,
    XPOS_KEY,
    YPOS_KEY,
    base_output_dir,
    get_output_path,
)
from mirar.processors.base_processor import BaseSourceProcessor

logger = logging.getLogger(__name__)


class RegionsWriter(BaseSourceProcessor):
    """
    Class to write a regions file from candidate table
    """

    base_key = "REGWRIT"

    def __init__(
        self,
        output_dir_name: Optional[str] = None,
        region_pix_radius: float = 8,
        output_dir: str | Path = base_output_dir,
    ):
        super().__init__()
        self.output_dir_name = output_dir_name
        self.region_pix_radius = region_pix_radius
        self.output_dir = Path(output_dir)

    def description(self) -> str:
        return (
            f"Processor to save 'region files' to "
            f"the {self.output_dir_name} directory. "
        )

    def _apply_to_sources(
        self,
        batch: SourceBatch,
    ) -> SourceBatch:
        for source_list in batch:
            candidate_table = source_list.get_data()

            regions_basepath = os.path.basename(source_list[BASE_NAME_KEY]).replace(
                ".fits", ".reg"
            )
            regions_path = get_output_path(
                regions_basepath,
                dir_root=self.output_dir_name,
                sub_dir=self.night_sub_dir,
                output_dir=self.output_dir,
            )

            regions_path.parent.mkdir(parents=True, exist_ok=True)

            with open(f"{regions_path}", "w", encoding="utf8") as regions_f:
                regions_f.write("image\n")
                for _, row in candidate_table.iterrows():
                    regions_f.write(
                        f"CIRCLE({row[XPOS_KEY]},{row[YPOS_KEY]},"
                        f"{self.region_pix_radius})\n"
                    )

        return batch
