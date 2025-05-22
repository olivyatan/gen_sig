import os
import pathlib
import shutil
import tempfile
from pathlib import PosixPath

import bxkrylov
import pyspark
import pytest
from pyspark.sql import functions as F
from pyspark.sql import types as T

DATA_ROOT = None


@pytest.fixture(scope="session")
def tests_root_dir():
    self_path = os.path.dirname(
        os.path.abspath(__file__)
    )
    return pathlib.Path(self_path)


@pytest.fixture(scope="session")
def data_dir(tests_root_dir):
    return os.path.join(tests_root_dir, "data")


@pytest.fixture(scope="session")
def regression_dir(tests_root_dir):
    return str(tests_root_dir / "regression")


def delete_spark_warehouse(tests_dir: PosixPath) -> None:
    parent_dir = os.path.dirname(tests_dir)
    shutil.rmtree(os.path.join(tests_dir, "spark-warehouse"), ignore_errors=True)
    shutil.rmtree(os.path.join(tests_dir, "metastore_db"), ignore_errors=True)
    shutil.rmtree(os.path.join(parent_dir, "spark-warehouse"), ignore_errors=True)
    shutil.rmtree(os.path.join(parent_dir, "metastore_db"), ignore_errors=True)


@pytest.fixture(scope="session")
def spark(request, tests_root_dir, data_dir):
    delete_spark_warehouse(tests_root_dir)
    active_session = pyspark.sql.SparkSession.getActiveSession()

    if active_session is not None:
        print(f"Active session app name = {active_session.sparkContext.appName}, "
              f"id = {active_session.sparkContext.applicationId}")
        active_session.stop()
    spark_session = bxkrylov.spark_session_for_test()
    print(
        f"Active session app name = {spark_session.sparkContext.appName}, id = {spark_session.sparkContext.applicationId}")

    # teardown function to clean up tables and Spark session
    def teardown():
        # Stop the Spark session
        spark_session.stop()

    # Add the teardown function to be called at the end of the session
    request.addfinalizer(teardown)

    spark_session.sql("CREATE DATABASE IF NOT EXISTS ACCESS_VIEWS")
    return spark_session


def check_map_col(df, col, new_map_id_col):
    df = df.selectExpr("*", f"explode({col}) as (key{new_map_id_col}, value{new_map_id_col})").drop(col)
    return df


def check_map_columns(expected_df, result_df):
    columns = list(expected_df.columns)
    for col in columns:
        new_map_id_col = 0
        if isinstance(expected_df.schema[col].dataType, T.MapType):
            expected_df = check_map_col(expected_df, col, new_map_id_col)
        if isinstance(result_df.schema[col].dataType, T.MapType):
            result_df = check_map_col(result_df, col, new_map_id_col)
        new_map_id_col += 1
    return expected_df, result_df


def assert_dataframe_equal(expected_df, result_df):
    assert len(set(expected_df.columns) ^ set(result_df.columns)) == 0, \
        f"Columns mismatch, expected_df: {expected_df.columns}, result_df: {result_df.columns}"

    expected_df, result_df = check_map_columns(expected_df, result_df)
    expected_df = expected_df.select(*sorted(expected_df.columns))
    result_df = result_df.select(*sorted(result_df.columns))

    delta = result_df.exceptAll(expected_df)
    if delta.count() != 0:
        print(f"Unexpected rows in result: {delta.collect()}")
        assert False
    delta = expected_df.exceptAll(result_df)
    if delta.count() != 0:
        print(f"Those rows were missing from result: {delta.collect()}")
        assert False
