import argparse
import os
import ssl
import subprocess
import threading
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Generator, List

import numpy as np
import requests
import tenacity
from pandas import DataFrame
from pyspark.sql import SparkSession

ENV_TO_API_URL = {'prod': 'https://recsmercury.vip.ebay.com',
                  'pre-prod': 'https://ozinman.recsmercury.pp.vip.ebay.com',
                  'local': 'http://localhost:8080'}
BATCH_SIZE = 300
NUM_THREADS = 1  # Number of parallel threads


def adjust_features(features):
    if features is None:
        return features
    elif isinstance(features, list):
        # Convert list of tuples to a dict string
        features = str(dict(features))
    elif isinstance(features, dict):
        # Convert dict to a string
        features = str(features)
    elif isinstance(features, str):
        # If it's already a string, return it as is
        return features
    else:
        raise ValueError(f"Unsupported type for features: {type(features)}")
    return features


def adjust_df_to_mercury_format(dfs: Generator[DataFrame, None, None], processor,
                                force: bool) -> Generator[tuple[int, list[dict]], None, None]:
    # input item:
    # Row(item_id=1, desc='raw_description', aspects=[('aspect1', 'value1'), ('aspect2', 'value2')], title='title1'),
    # output item:
    # {"processor": "GenSignalsV1", "itemId": 276374983729, "title": "title",
    #  "features": '{"aspect1": "value1", "aspect2": "value2"}",
    #  "rawDescription": "rawDescription"}
    for i, df in enumerate(dfs):
        df = df.rename(columns={"item_id": "itemId", "desc": "rawDescription", "aspects": "features"})
        df['itemId'] = df['itemId'].astype(int)
        # convert list of tuples to a dict string
        df['features'] = df['features'].apply(adjust_features)
        df['processor'] = processor
        if force:
            df['force'] = True
        batch_list = df.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
        yield i, batch_list


def split_dataframe(df, batch_size):
    num_chunks = len(df) // batch_size + 1
    for i in range(num_chunks):
        yield df[i * batch_size: (i + 1) * batch_size]


def read_parquet_in_chunks(spark, hdfs_path) -> Generator[DataFrame, None, None]:
    spark_df = spark.read.parquet(hdfs_path)
    pandas_df = spark_df.toPandas()
    for batch_df in split_dataframe(pandas_df, BATCH_SIZE):
        yield batch_df  # noqa


def wrapped_post(api_url: str, batch_list: List[Dict[Any, Any]], i: int, thread_name: str):
    """
    Send data to API with retries and batch splitting logic.
    Will retry on SSL errors and split batches on size limit errors.
    """

    @tenacity.retry(
        retry=(
            tenacity.retry_if_exception_type(ssl.SSLEOFError)
            | tenacity.retry_if_exception_type(requests.exceptions.ConnectionError)
        ),
        wait=tenacity.wait_exponential(multiplier=1.5, min=2),
        stop=tenacity.stop_after_attempt(10),  # almost 60 seconds
        before_sleep=lambda retry_state: print(
            f"{thread_name} - Retry attempt {retry_state.attempt_number} after error: {retry_state.outcome and retry_state.outcome.exception()}")
    )
    def post_with_retry(url: str, data: List[Dict[Any, Any]]):
        _response = requests.post(url, json=data, verify=False)

        # If we get a 500 error with message size limit, split and retry
        if _response.status_code == 500 and "message exceeds maximum" in _response.text.lower():
            if len(data) <= 1:
                print(f"{thread_name} - Cannot split batch further, single record too large")
                _response.raise_for_status()

            mid = len(data) // 2
            first_half = data[:mid]
            second_half = data[mid:]

            print(f"{thread_name} - Splitting batch of size {len(data)} into {len(first_half)} and {len(second_half)}")

            post_with_retry(url, first_half)
            return post_with_retry(url, second_half)

        _response.raise_for_status()
        return _response

    print(f"{thread_name} - Sending batch {i} (size: {len(batch_list)})")
    response = post_with_retry(api_url, batch_list)
    print(f"{thread_name} - Sent batch {i} status_code: {response.status_code}")


def send_batches(enumerated_batch_list: tuple[int, list[dict]], api_url: str, route_to_machine: bool,
                 limit: int | None):
    thread_name = threading.current_thread().name
    i, batch_list = enumerated_batch_list
    try:
        if route_to_machine:
            for j, item in enumerate(batch_list):
                if limit is not None and j + i * BATCH_SIZE >= limit:
                    print(f"{thread_name} - Limit reached {i}, stopping further processing.")
                    return True
                _response = requests.post(api_url, json=item, verify=False)
                print(f"{thread_name} - Sent item {j} batch {i} status_code: {_response.status_code}")
        else:
            wrapped_post(api_url, batch_list, i, thread_name)
        return True
    except Exception as e:
        print(f"{thread_name} - Error sending batch {i}: {e}")
        return False


def get_api_url(env, route_to_machine):
    api_url = ENV_TO_API_URL[env]
    endpoint = "api/nrt/generative-signals/numsg/default/route" \
        if route_to_machine else "api/nrt/generative-signals/numsg/produce"
    return f"{api_url}/{endpoint}"


def run(spark, env, hdfs_path, processor, force: bool, route_to_machine: bool, limit: int | None):
    api_url = get_api_url(env, route_to_machine)

    batched_dfs: Generator[DataFrame, None, None] = read_parquet_in_chunks(spark, hdfs_path)
    enumerate_batch_generator = adjust_df_to_mercury_format(batched_dfs, processor, force)

    thread_pool = ThreadPool(NUM_THREADS)
    _send_batches = lambda enumerated_batch_list: send_batches(enumerated_batch_list, api_url, route_to_machine, limit)
    results = thread_pool.map(_send_batches, enumerate_batch_generator)

    if all(results):
        print("Data upload complete.")
    else:
        print("Some batches failed to upload. Please check the logs.")
        raise Exception("Some batches failed to upload. Please check the logs.")


def add_args():
    parser.add_argument("--hdfs-path", type=str, required=True,
                        help="Path to the HDFS file, on local env, it should be a local file path")
    parser.add_argument("--env", type=str, required=True, help="environment: [local, pre-prod, prod]")
    parser.add_argument("--processor", type=str, required=True,
                        help="processor in mercury to handle this item, e.g: GenSignalsV1", default="GenSignalsV1")
    parser.add_argument("--route-to-machine", action="store_true", required=False, default=False,
                        help="This will call the route API, forcing the item to be processed on the machine, "
                             "and not sent to Kafka where it can be processes by any machine")
    parser.add_argument("--force", action="store_true", required=False, default=False,
                        help="Force Mercury to process the items even if they were processed recently")
    parser.add_argument("--limit-items", type=int, required=False, default=None,
                        help="Works only with --route-to-machine, limit the number of items to be processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data to Mercury")
    add_args()
    args = parser.parse_args()
    if args.limit_items is not None and not args.route_to_machine:
        parser.error("--limit-items can only be used with --route-to-machine")

    spark = (
        SparkSession.builder.appName("bepersonalservice_generative_signals_load_to_mercury")
        .enableHiveSupport()
        .getOrCreate()
    )
    sc = spark.sparkContext
    log4jLogger = sc._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)

    logger.info("Starting loading items to mercury")
    run(spark, args.env, args.hdfs_path, args.processor, args.force, args.route_to_machine, args.limit_items)
    logger.info("Finished loading items to mercury")
