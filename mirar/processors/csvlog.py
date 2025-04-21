"""
Module to generate a CSV log of observations
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from mirar.data import ImageBatch
from mirar.paths import BASE_NAME_KEY, core_fields, get_output_path
from mirar.processors.base_processor import BaseImageProcessor

logger = logging.getLogger(__name__)

default_log_keys = [BASE_NAME_KEY] + core_fields


class CSVLog(BaseImageProcessor):
    """
    Processor to generate a CSV log
    """

    max_n_cpu = 1

    base_key = "csvlog"

    def __init__(
        self,
        export_keys: Optional[list[str]] = None,
        output_sub_dir: str = "",
        output_base_dir: Optional[str] = None,
        output_name: str = "log",
    ):
        super().__init__()
        if export_keys is None:
            export_keys = default_log_keys
        self.export_keys = export_keys
        self.output_sub_dir = output_sub_dir
        self.output_base_dir = output_base_dir
        self.output_name = output_name
        self.all_rows = []

    def description(self) -> str:
        return (
            f"Create a CSV log with {len(self.export_keys)} columns, "
            f"named {self.get_log_name()}"
        )

    def get_log_name(self) -> str:
        """
        Returns the custom log name

        :return: Log file name
        """
        return f"{self.night}_{self.output_name}.csv"

    def get_output_path(self) -> Path:
        """
        Returns the full log output path

        :return: log path
        """
        output_base_dir = self.output_base_dir
        if output_base_dir is None:
            output_base_dir = self.night_sub_dir

        output_path = get_output_path(
            base_name=self.get_log_name(),
            dir_root=output_base_dir,
            sub_dir=self.output_sub_dir,
        )

        try:
            os.makedirs(os.path.dirname(output_path))
        except OSError:
            pass

        return output_path

    def _apply_to_images(
        self,
        batch: ImageBatch,
    ) -> ImageBatch:
        output_path = self.get_output_path()

        # One row in log per batch
        row = []
        for key in self.export_keys:
            row.append(batch[0][key])
        self.all_rows.append(row)

        # Update log
        log = pd.DataFrame(self.all_rows, columns=self.export_keys)
        logger.debug(f"Saving log with {len(log)}  rows to: {output_path}")
        log.to_csv(output_path)

        return batch
