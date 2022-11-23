import pandas as pd
from winterdrp.data.base_data import DataBlock, DataBatch


class SourceTable(DataBlock):

    def __init__(
            self,
            source_list: pd.DataFrame,
    ):

        self.source_list = source_list
        self.metadata = dict()

    def get_data(self) -> pd.DataFrame:
        return self.source_list

    def set_data(self, source_list: pd.DataFrame):
        self.source_list = source_list

    def get_metadata(self) -> dict:
        return self.metadata

    def __getitem__(self, item):
        return self.metadata.__getitem__(item)

    def __setitem__(self, key, value):
        self.metadata.__setitem__(key, value)

    def keys(self):
        return self.metadata.keys()


class SourceBatch(DataBatch):

    def __add__(self, data: SourceTable):
        self._batch.append(data)

    def get_batch(self) -> list[SourceTable]:
        return self._batch