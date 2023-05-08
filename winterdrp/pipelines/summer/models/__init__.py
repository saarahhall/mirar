"""
Models for database and pydantic dataclass models
"""
from typing import Union

from sqlalchemy.orm import DeclarativeBase

from winterdrp.pipelines.summer.models._exposures import Exposure, ExposuresTable
from winterdrp.pipelines.summer.models._fields import (
    FieldEntry,
    FieldsTable,
    populate_fields,
)
from winterdrp.pipelines.summer.models._filters import (
    Filter,
    FiltersTable,
    populate_filters,
)
from winterdrp.pipelines.summer.models._img_type import (
    ALL_ITID,
    ImgType,
    ImgTypesTable,
    populate_itid,
)
from winterdrp.pipelines.summer.models._nights import Night, NightsTable
from winterdrp.pipelines.summer.models._proc import Proc, ProcTable
from winterdrp.pipelines.summer.models._programs import (
    Program,
    ProgramCredentials,
    ProgramsTable,
    default_program,
    populate_programs,
)
from winterdrp.pipelines.summer.models._raw import Raw, RawTable
from winterdrp.pipelines.summer.models._subdets import (
    SubDet,
    SubdetsTable,
    populate_subdets,
)
from winterdrp.pipelines.summer.models.base_model import SummerBase
from winterdrp.processors.sqldatabase.base_model import BaseTable
from winterdrp.processors.sqldatabase.postgres import PostgresAdmin
from winterdrp.processors.sqldatabase.postgres_utils import (
    ADMIN_PASSWORD,
    ADMIN_USER,
    DB_PASSWORD,
    DB_USER,
)
from winterdrp.utils.sql import get_engine


def setup_database(base: Union[DeclarativeBase, BaseTable]):
    """
    Function to setup database
    Args:
        base:
    Returns:
    """
    if DB_USER is not None:
        db_name = base.db_name
        admin_engine = get_engine(
            db_name=db_name, db_user=ADMIN_USER, db_password=ADMIN_PASSWORD
        )

        pg_admin = PostgresAdmin()

        if not pg_admin.check_if_db_exists(db_name=db_name):
            pg_admin.create_db(db_name=db_name)

        base.metadata.create_all(
            admin_engine
        )  # extensions need to be created as a superuser

        if not pg_admin.check_if_user_exists(user_name=DB_USER):
            pg_admin.create_new_user(new_db_user=DB_USER, new_password=DB_PASSWORD)

        pg_admin.grant_privileges(db_name=db_name, db_user=DB_USER)


if DB_USER is not None:
    setup_database(SummerBase)
    populate_fields()
    populate_itid()
    populate_filters()
    populate_programs()
    populate_subdets()