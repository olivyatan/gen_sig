import logging
from argparse import ArgumentParser
from datetime import datetime, timedelta

import pyspark.sql
from bxkrylov.javaudf import jarbuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.sql.utils import AnalysisException

VACANT_SLOT_RANK = 999999999

LIMIT_ITEMS_FACTOR = 4  # we factor limit items at the start and then limit to the exact number at the final stage

logging.getLogger().setLevel(logging.ERROR)


PREVIOUSLY_PROCESSED_ITEMS_PATH = '/apps/b_perso/generative_signals/previously_items_processed.parquet/'
VALID_ASPECT_TYPES = ['df']
INVALID_ASPECT_VALUES = ["", "null", "none", "na", "n/a", "no", "unknown", "false", "unbranded", "other"]
INVALID_ASPECT_NAMES = ['mergednamespacenames', 'maturityscore', 'condition', 'upca+2',
                        'isprplinkenabled', 'lhexposetotse', 'seller-selected-epid',
                        'non-domestic product', 'modified item', 'upc', 'mergednamespacenames'
                        'producttitle', 'ean', 'savedautotagprodrefid', 'ebay product id (epid)',
                        'gtin', 'california prop 65 warning', 'catrecoscore_1',
                        'catrecoscore_2', 'catrecoscore_3', 'catrecoid_1', 'catrecoid_2',
                        'catrecoid_3', 'miscatscore', 'p2sprobability',
                        'uom1', 'miscatscore_v1', 'uom2', 'mpn', 'uom3', 'productimagezoomguid',
                        'manufacturer part number', 'isbn-13', 'isbn-10', 'other part number',
                        'miscatscore_cf_v1', 'isclplinkenabled', 'oe/oem part number',
                        'features', 'model', 'ks', 'number in pack', 'style code', 'productimagezoomguid',
                        'item height', 'item length', 'item width', 'item weight', 'isprplinkenabled',
                        'number of items in set', 'food aisle', 'width', 'length', 'items included',
                        'custom bundle', 'volume', 'period after opening (pao)', 'featured refinements',
                        'set includes', 'catrecoid', 'catrecoscore', 'core_product_type_v2',
                        'miscatscore', 'mergednamespacenames' , 'miscatscore', 'miscatflag_v1',
                        'catrecoid', 'catrecoscore', 'core_product_type_v2', 'miscatscore', 'mergednamespacenames']


INVALID_ASPECT_SRC = [287, 227, 160, 42, 104, 158, 39, 37, 30, 22, 6, 2]


def read_shown_signals(field):
    return F.expr(f"case when coalesce(bxsoj_nvl(soj, '{field}')) is not null then "
                  rf"split(bxsoj_url_decode_repeat(coalesce(bxsoj_nvl(soj, '{field}'))), '[,|\\[\\]]') else null end") \
        .cast(ArrayType(StringType()))


def get_force_decode_element_from_soj(field):
    return F.expr(f"case when bxsoj_nvl(soj, '{field}') is not null then bxsoj_url_decode_repeat_force_once(bxsoj_nvl(soj, '{field}')) else null end")


def get_eligible_viewed_item_ids(spark: pyspark.sql.SparkSession, start_date: str, end_date: str,
                                 limit_items: int):
    df = get_relevant_viewed_items(spark, start_date, end_date)
    agg_df = df.groupBy('item_id').agg(
        F.last('site_id').alias('site_id'),
        F.last('leaf_categ_id').alias('leaf_categ_id'),
        F.min('signals_shown_conversational_size').alias('signals_shown_conversational_size'),
        F.count("*").alias('cnt')
    )
    agg_df = agg_df.withColumn('rank', F.expr(f'case when signals_shown_conversational_size < 2 then {VACANT_SLOT_RANK} else cnt end'))
    agg_df = agg_df.select('item_id', 'site_id', 'leaf_categ_id', 'rank')
    filtered_df = agg_df.orderBy("rank", ascending=False).limit(limit_items * LIMIT_ITEMS_FACTOR)
    return filtered_df.cache()


def get_relevant_viewed_items(spark, start_date, end_date):
    df = spark.read.table("ACCESS_VIEWS.VI_EVENT_FACT").where(
        (F.col('session_start_dt').between(start_date, end_date))
        & (F.col('item_id') > 0)
        & (F.col('exclude').eqNullSafe(0)))
    df = df.select('item_id', 'site_id', 'soj', 'leaf_categ_id')
    df = df.withColumn('signals_shown_conversational_size', F.size(read_shown_signals('viwtbidstr')))
    return df


def filter_items_relevant_cats(items_df, cats_table):
    items_df = items_df.join(
        F.broadcast(cats_table),
        on=((items_df['leaf_categ_id'] == cats_table['leaf_categ_id'])
            & (items_df['SITE_ID'] == cats_table['SITE_ID'])),
        how='left_semi'
    )
    return items_df.select('item_id')


def get_categ_df(spark, business_vertical_names, excluded_categ_lvl3_id, site_ids):
    categ_df = spark.table('ACCESS_VIEWS.DW_CATEGORY_GROUPINGS') \
        .filter(F.col('leaf_categ_id') == F.col('MOVE_TO'))  # takes the most update leaf category
    if site_ids is not None:
        categ_df = categ_df.filter(F.col('site_id').isin(site_ids))
    # Exclude specific leaf categories if provided
    if excluded_categ_lvl3_id is not None:
        categ_df = categ_df.filter(~F.col('categ_lvl3_id').isin(excluded_categ_lvl3_id))
    if business_vertical_names is not None:
        categ_df = categ_df.filter(F.col('bsns_vrtcl_name').isin(business_vertical_names))
    return categ_df


def get_broadcast_item_ids_df(items_ids_df):
    items_ids_df = items_ids_df.select('item_id').distinct()
    return F.broadcast(items_ids_df)


def get_previously_processes_item_ids_df(spark, start_date: str):
    schema = StructType([StructField('item_id', IntegerType(), True)])
    year_ago = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    return (spark.read.schema(schema).parquet(PREVIOUSLY_PROCESSED_ITEMS_PATH)
            .filter(F.col('session_start_dt') >= year_ago))


def filter_previously_processed_item_ids(spark, broadcast_items_ids_df: DataFrame, start_date: str,
                                         is_prod: bool):
    if is_prod:
        previously_processes_item_ids = get_previously_processes_item_ids_df(spark, start_date)
        return F.broadcast(broadcast_items_ids_df.join(previously_processes_item_ids, 'item_id', 'left_anti'))
    else:
        return broadcast_items_ids_df


def get_items_data(spark, broadcast_items_ids_df: DataFrame):
    listing_df: DataFrame = (spark.table("ACCESS_VIEWS.DW_LSTG_ITEM").select('ITEM_ID', 'AUCT_TITL', 'LSTG_STATUS_ID')
                             .withColumnRenamed('AUCT_TITL', 'title'))
    relevant_listing_df = listing_df.join(broadcast_items_ids_df, 'ITEM_ID', 'left_semi')
    filtered_df = relevant_listing_df.filter(F.col('LSTG_STATUS_ID') == 0)  # live items
    return filtered_df.select('item_id', 'title')


def get_items_desc(spark, broadcast_new_items_ids_df):
    description_df = spark.table("ACCESS_VIEWS.LSTG_ITEM_DESC_DIM") \
        .select('item_id', 'item_desc').withColumnRenamed('item_desc', 'desc')
    return description_df.join(broadcast_new_items_ids_df, 'item_id', 'left_semi')


def lowercase_string_columns(df, columns):
    for col_name in columns:
        df = df.withColumn(col_name, F.lower(F.col(col_name)))
    return df


def get_items_aspects(spark, broadcast_new_items_ids_df, start_date):
    aspects_df = spark.table("ACCESS_VIEWS.ITEM_ASPCT_CLSSFCTN_SAP").filter((F.col('auct_end_dt') >= start_date))
    relevant_items_aspects_df = aspects_df.join(broadcast_new_items_ids_df, 'item_id', 'left_semi')

    filtered_aspects_df = relevant_items_aspects_df.filter(
        (F.col('NS_TYPE_CD').isin(VALID_ASPECT_TYPES))
        & (~F.lower(F.col('PRDCT_ASPCT_NM')).isin(INVALID_ASPECT_NAMES))
        & (~F.col('ASPCT_SRC').isin(INVALID_ASPECT_SRC))
        & (F.col('ASPCT_VLU_NM').isNotNull())
        & (~F.lower(F.col('ASPCT_VLU_NM')).isin(INVALID_ASPECT_VALUES))
    )

    lowered_filtered_aspects_df = lowercase_string_columns(filtered_aspects_df, ['PRDCT_ASPCT_NM', 'ASPCT_VLU_NM'])
    result_df = (
        lowered_filtered_aspects_df.groupBy('item_id')
        .agg(F.map_from_entries(F.collect_list(F.struct(F.col('PRDCT_ASPCT_NM'), F.col('ASPCT_VLU_NM')))).alias('aspects'))
    )

    return result_df


def persist_final_df(final_df, output_path):
    if output_path:
        final_df.write.parquet(output_path, mode="overwrite", compression="snappy")


def persist_items_to_previously_processed(broadcast_new_items_ids_df, start_date, is_prod):
    if is_prod:
        broadcast_new_items_ids_df.write.parquet(f'{PREVIOUSLY_PROCESSED_ITEMS_PATH}session_start_dt={start_date}/',
                                                 mode="append", compression="snappy")


def validate_output_path_is_empty(spark, output_path):
    try:
        df = spark.read.parquet(output_path)
        has_data = df.rdd.isEmpty() == False
    except AnalysisException:
        has_data = False  # Path doesn't exist or can't be read
    if has_data:
        raise Exception(f"Output path {output_path} is not empty. Please provide an empty path.")


def handle_limit_by_rank(all_data_df, filtered_viewed_items_df, limit_items):
    selected_filtered_viewed_items_df = filtered_viewed_items_df.select('item_id', 'rank')
    final_with_viewed_data_df = all_data_df.join(selected_filtered_viewed_items_df, on='item_id', how='inner').orderBy('rank', ascending=False)
    final_ranked_df = final_with_viewed_data_df.limit(limit_items)
    return final_ranked_df


def extract_items_for_generation(spark: pyspark.sql.SparkSession, start_date: str, end_date: str, output_path: str,
                                 is_prod: bool, business_vertical_names: list[str] | None = None,
                                 site_ids: list[int] | None = None, excluded_categ_lvl3_id: list[int] | None = None,
                                 limit_items: int = 4_320_000) -> DataFrame:
    """
    Extracts items for generation based on the specified criteria and time range.

    This function filters and processes item data from various tables, including item descriptions and aspects,
    and persists the final DataFrame to the specified output path. It also handles previously processed items
    to avoid duplication.

    Parameters:
    - spark (pyspark.sql.SparkSession): The Spark session to use for data processing.
    - start_date (str): The start date for filtering viewed items (inclusive).
    - end_date (str): The end date for filtering viewed items (inclusive).
    - output_path (str): The path where the final DataFrame will be persisted.
    - is_prod (bool): if running in production mode, it will persist the processed items to the
        "previously processed items path". Saving will cause those items to be excluded from future processing.
    - business_vertical_names (list, optional): List of business vertical names to filter categories.
    - site_ids (list, optional): List of site IDs to filter items.
    - excluded_categ_lvl3_id (list, optional): List of Level 3 category IDs to exclude.
    - limit_items (int): The maximum number of items to include in the final DataFrame.
        default is 4,320,000. which is the expected TPS of 50 items/sec 24 hours.

    Returns:
    - DataFrame: The final DataFrame containing the filtered and processed item data.
    """
    spark.conf.set("spark.sql.mapKeyDedupPolicy", "LAST_WIN")  # to avoid duplicate keys in aspect map
    validate_output_path_is_empty(spark, output_path)

    viewed_items_df = get_eligible_viewed_item_ids(spark, start_date, end_date, limit_items)
    categ_df = get_categ_df(spark, business_vertical_names, excluded_categ_lvl3_id, site_ids)
    filtered_viewed_items_df = filter_items_relevant_cats(viewed_items_df, categ_df)
    broadcast_items_ids_df = get_broadcast_item_ids_df(filtered_viewed_items_df)
    broadcast_new_items_ids_df = filter_previously_processed_item_ids(spark, broadcast_items_ids_df, start_date,
                                                                      is_prod)
    items_data = get_items_data(spark, broadcast_new_items_ids_df)
    desc_df = get_items_desc(spark, broadcast_new_items_ids_df)
    aspects_df = get_items_aspects(spark, broadcast_new_items_ids_df, start_date)
    all_data_df = items_data.join(aspects_df, 'item_id', 'left').join(desc_df, 'item_id', 'left')
    all_data_df = all_data_df.filter(F.col('desc').isNotNull() | F.col('aspects').isNotNull())

    final_ranked_df = handle_limit_by_rank(all_data_df, viewed_items_df, limit_items)
    final_ranked_df.cache()
    total_count = final_ranked_df.count()
    vacant_slot_count = final_ranked_df.filter(F.col('rank') == VACANT_SLOT_RANK).count()
    print(f'Final DF stats: Total count: {total_count}, vacant slot count: {vacant_slot_count} '
          f'non-vacant slot count: {total_count - vacant_slot_count} limit items: {limit_items}')
    final_df = final_ranked_df.drop('rank')

    persist_final_df(final_df, output_path)
    persist_items_to_previously_processed(broadcast_new_items_ids_df, start_date, is_prod)
    return final_df


def add_args(parser):
    parser.add_argument(
        "--output-path",
        help="Path to output results to",
        default="viewfs://apollo-rno/apps/b_perso/generative_signals/new_items/",
        required=True,
    )
    parser.add_argument(
        "--start-date",
        help="Input date ex. 2021-04-07",
        required=False,
    )
    parser.add_argument(
        "--end-date",
        help="Input date ex. 2021-04-07",
        required=False,
    )
    parser.add_argument(
        "--run-date",
        help="Input date ex. 2021-04-07",
        required=False,
    )
    parser.add_argument(
        "--is-prod",
        help="Is this a production run? If true, it will persist the processed items to the previously processed items path.",
        default=False,
    )
    parser.add_argument(
        "--business-vertical-names",
        help="Business vertical names to filter categories",
        nargs='+',
        default=['Fashion', 'Home & Garden']
    )
    parser.add_argument(
        "--site-ids",
        help="Site IDs to filter items",
        nargs='+',
        type=int,
        default=[0]
    )
    parser.add_argument(
        "--excluded-categ-lvl3-id",
        help="Level 3 category IDs to exclude",
        nargs='+',
        type=int,
        default=[176992]
    )
    parser.add_argument(
        "--limit-items",
        help="Maximum number of items to include in the final DataFrame",
        type=int,
        default=None
    )


def main():
    parser = ArgumentParser(description="Generative signals item selection")
    add_args(parser)
    args = parser.parse_args()
    spark = (
        SparkSession.builder.appName("bepersonalservice_generative_signals_item_selection")
        .enableHiveSupport()
        .getOrCreate()
    )
    jarbuilder.java_register_internal_udfs(spark)
    sc = spark.sparkContext
    log4jLogger = sc._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)
    logger.info("Starting extracting items for generation")
    if args.start_date is None and args.end_date is None and args.run_date:
        start_date = args.run_date
        end_date = args.run_date
    elif args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        raise Exception("Either start_date and end_date or run_date must be provided")
    extract_items_for_generation(spark, start_date, end_date, args.output_path, args.is_prod,
                                 args.business_vertical_names, args.site_ids, args.excluded_categ_lvl3_id,
                                 args.limit_items)
    logger.info("Finished extracting items for generation")


if __name__ == '__main__':
    main()
