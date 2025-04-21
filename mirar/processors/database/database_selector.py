"""
Module containing processors which import values from a database
"""

import logging
from abc import ABC
from collections.abc import Callable
from typing import Optional

import pandas as pd

from mirar.data import DataBlock, Image, ImageBatch, SourceBatch
from mirar.database.constraints import DBQueryConstraints
from mirar.database.transactions import select_from_table
from mirar.paths import SOURCE_HISTORY_KEY
from mirar.processors.base_processor import BaseImageProcessor, BaseSourceProcessor
from mirar.processors.database.base_database_processor import BaseDatabaseProcessor

logger = logging.getLogger(__name__)


class BaseDatabaseSelector(BaseDatabaseProcessor, ABC):
    """Base Class for any database selector"""

    base_key = "dbselector"

    def __init__(
        self,
        *args,
        boolean_match_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.boolean_match_key = boolean_match_key

    def get_constraints(self, data: dict) -> None | DBQueryConstraints:
        """
        Get db query constraints for a given datablock object

        :param data: data block
        :return: db query constraints object
        """
        raise NotImplementedError()


def update_header_with_single_match(data: DataBlock, res: list[dict]) -> DataBlock:
    """
    Update a datablock with a single db query result

    :param data: datablock to update
    :param res: corresponding db query
    :return: updated datablock
    """
    assert len(res) == 1

    for key, value in res[0]:
        data[key] = value

    return data


class BaseImageDatabaseSelector(BaseDatabaseSelector, BaseImageProcessor, ABC):
    """
    Processor to import data from images
    """

    def __init__(
        self,
        db_output_columns: str | list[str],
        output_alias_map: Optional[str | list[str]] = None,
        update_header: Callable[
            [Image, list[dict]], Image
        ] = update_header_with_single_match,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.update_header = update_header
        self.db_output_columns = db_output_columns
        self.output_alias_map = output_alias_map

    def _apply_to_images(
        self,
        batch: ImageBatch,
    ) -> ImageBatch:
        for i, image in enumerate(batch):
            query_constraints = self.get_constraints(image)

            res = select_from_table(
                sql_table=self.db_table,
                db_constraints=query_constraints,
                output_columns=self.db_output_columns,
            )

            image = self.update_header(image, res)

            if self.boolean_match_key is not None:
                image[self.boolean_match_key] = len(res) > 0

            batch[i] = image

        return batch


class BaseValuesCrossmatch(BaseDatabaseSelector, ABC):
    """Processor to crossmatch to a database"""

    def __init__(self, db_query_columns: str | list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_query_columns = db_query_columns

    def get_accepted_values(self, data: dict) -> list[str | float | int]:
        """
        Get list of accepted values for crossmatch query

        :param data: datablock
        :return: accepted values from datablock
        """
        accepted_values = [data[x.upper()] for x in self.db_query_columns]
        return accepted_values

    def get_constraints(self, data: dict) -> DBQueryConstraints:
        """
        Get db query constraints for a datablock

        :param data: datablock
        :return: list of constraints
        """
        query_columns = self.db_query_columns
        accepted_values = self.get_accepted_values(data)
        comparison_types = ["=" for _ in self.db_query_columns]
        query_constraints = DBQueryConstraints(
            columns=query_columns,
            accepted_values=accepted_values,
            comparison_types=comparison_types,
        )
        return query_constraints


class CrossmatchDatabaseWithHeader(BaseImageDatabaseSelector, BaseValuesCrossmatch):
    """
    Processor to crossmatch to a database using keys
    """

    def description(self) -> str:
        return f"Crossmatch to database using keys {self.db_query_columns}"


class BaseDatabaseSourceSelector(BaseDatabaseSelector, BaseSourceProcessor, ABC):
    """
    Base Class for dataframe DB importers
    """

    def __init__(
        self,
        db_output_columns: str | list[str],
        max_num_results: Optional[int] = None,
        additional_query_constraints: DBQueryConstraints | None = None,
        **kwargs,
    ):
        self.db_output_columns = db_output_columns
        self.max_num_results = max_num_results
        self.additional_query_constraints = additional_query_constraints
        super().__init__(**kwargs)

    def update_dataframe(
        self, candidate_table: pd.DataFrame, results: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Update a dataframe with db results

        :param candidate_table: pandas table
        :param results: results from db query
        :return: updated dataframe
        """
        raise NotImplementedError()

    def _apply_to_sources(
        self,
        batch: SourceBatch,
    ) -> SourceBatch:
        for source_table in batch:
            metadata = source_table.get_metadata()
            candidate_table = source_table.get_data()
            results = []
            for _, source in candidate_table.iterrows():

                res = self.query_for_source(source, metadata)

                results.append(res)

            new_table = self.update_dataframe(candidate_table, results)
            source_table.set_data(new_table)
        return batch

    def query_for_source(self, source: pd.Series, metadata: dict) -> pd.DataFrame:
        """
        Query the database for a single source

        :param source: Source data
        :param metadata: Source Batch metadata
        :return: Results from the database
        """
        super_dict = self.generate_super_dict(metadata, source)
        query_constraints = self.get_constraints(super_dict)
        logger.debug(f"Query constraints: " f"{query_constraints.parse_constraints()}")
        if self.additional_query_constraints is not None:
            query_constraints = query_constraints + self.additional_query_constraints
        logger.debug(f"Query constraints: " f"{query_constraints.parse_constraints()}")
        res = select_from_table(
            sql_table=self.db_table.sql_model,
            db_constraints=query_constraints,
            output_columns=self.db_output_columns,
            max_num_results=self.max_num_results,
        )
        return res


class DatabaseSingleMatchSelector(BaseDatabaseSourceSelector, ABC):
    """
    Processor to import a single match from a database
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_num_results=1, **kwargs)

    def update_dataframe(
        self, candidate_table: pd.DataFrame, results: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Update a dataframe with db results

        :param candidate_table: pandas table
        :param results: results from db query
        :return: updated dataframe
        """
        assert len(results) == len(candidate_table)

        new_cols = []
        for res in results:
            if len(res) > 0:
                assert len(res) == 1
                new_row = {x: res.iloc[0][x] for x in self.db_output_columns}

            else:
                new_row = {x: None for x in self.db_output_columns}

            new_cols.append(new_row)

        candidate_table = candidate_table.join(pd.DataFrame(new_cols))

        return candidate_table


class DatabaseMultimatchSelector(BaseDatabaseSourceSelector, ABC):
    """
    Processor to import multiple matches from a database
    """

    def __init__(self, *args, base_output_column: str = SOURCE_HISTORY_KEY, **kwargs):
        self.base_output_column = base_output_column
        super().__init__(*args, **kwargs)

    def update_dataframe(
        self,
        candidate_table: pd.DataFrame,
        results: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Update a pandas dataframe with the number of matches

        :param candidate_table: Pandas dataframe
        :param results: db query results
        :return: updated pandas dataframe
        """
        assert len(results) == len(candidate_table)
        candidate_table[self.base_output_column] = results
        return candidate_table


class BaseSpatialCrossmatchSource(BaseDatabaseSourceSelector, ABC):
    """
    Processor to crossmatch to sources in a database using spatial search
    """

    def __init__(
        self,
        crossmatch_radius_arcsec: float,
        ra_field_name: str = "ra",
        dec_field_name: str = "dec",
        order_field_name: Optional[str] = None,
        order_ascending: bool = False,
        query_dist: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.xmatch_radius_arcsec = crossmatch_radius_arcsec
        self.ra_field_name = ra_field_name
        self.dec_field_name = dec_field_name
        self.order_field_name = order_field_name
        self.order_ascending = order_ascending
        self.query_dist = query_dist

    def get_source_crossmatch_constraints(self, data: dict) -> DBQueryConstraints:
        """
        Apply constraints to a single source, using q3c

        :param data: Dictionary containing source data
        :return: DBQueryConstraints
        """
        query_constraints = DBQueryConstraints()
        query_constraints.add_q3c_constraint(
            ra=data["ra"],
            dec=data["dec"],
            ra_field_name=self.ra_field_name,
            dec_field_name=self.dec_field_name,
            crossmatch_radius_arcsec=self.xmatch_radius_arcsec,
        )

        return query_constraints

    def get_constraints(self, data: dict) -> DBQueryConstraints:
        return self.get_source_crossmatch_constraints(data)


class SingleSpatialCrossmatchSource(
    BaseSpatialCrossmatchSource, DatabaseSingleMatchSelector
):
    """
    Processor to import a single source from a database using spatial crossmatch
    """

    def description(self) -> str:
        return f"Crossmatch to db using radius {self.xmatch_radius_arcsec}, limit 1"


class SpatialCrossmatchSourceWithDatabase(
    BaseSpatialCrossmatchSource, DatabaseMultimatchSelector
):
    """
    Processor to import multiple sources from a database using spatial crossmatch
    """

    def description(self) -> str:
        return f"Crossmatch to db using radius {self.xmatch_radius_arcsec}"


class SelectSourcesWithMetadata(DatabaseMultimatchSelector, BaseValuesCrossmatch):
    """
    Processor to import sources from a database using metadata values
    """

    def description(self) -> str:
        return f"Crossmatch to db using keys {self.db_query_columns}"


class DatabaseHistorySelector(SpatialCrossmatchSourceWithDatabase):
    """
    Processor to import previous detections of a source from a database
    """

    def __init__(
        self,
        history_duration_days: float,
        time_field_name: str = "jd",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.history_duration_days = history_duration_days
        self.time_field_name = time_field_name
        self.output_df_colname = SOURCE_HISTORY_KEY
        logger.debug(f"Update db is {self.update_dataframe}")

    def get_constraints(self, data: dict) -> DBQueryConstraints:
        query_constraints = self.get_source_crossmatch_constraints(data)
        query_constraints.add_constraint(
            column=self.time_field_name,
            comparison_type="<",
            accepted_values=data[self.time_field_name],
        )
        query_constraints.add_constraint(
            column=self.time_field_name,
            comparison_type=">=",
            accepted_values=data[self.time_field_name] - self.history_duration_days,
        )
        return query_constraints
