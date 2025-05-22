from unittest import mock

from pyspark.sql import Row

from generative_signals.etl.item_selection import (
    INVALID_ASPECT_SRC,
    INVALID_ASPECT_VALUES,
    VACANT_SLOT_RANK,
    VALID_ASPECT_TYPES,
    extract_items_for_generation,
    filter_items_relevant_cats,
    filter_previously_processed_item_ids,
    get_eligible_viewed_item_ids,
    get_items_aspects,
)
from generative_signals.tests.conftest import assert_dataframe_equal


def test_get_eligible_viewed_item_ids(spark):
    session_start_dt = '2025-03-05'
    site_id = 0
    leaf_categ_id = 100
    vi_data = [
        Row(item_id=1, soj='soj1', site_id=site_id, leaf_categ_id=leaf_categ_id,
            session_start_dt=session_start_dt, exclude=0),
        Row(item_id=1, soj='soj1', site_id=site_id, leaf_categ_id=leaf_categ_id,
            session_start_dt=session_start_dt, exclude=0),  # duplicate
        Row(item_id=2, soj='dummy=dummy&viwtbidstr=', site_id=site_id, leaf_categ_id=leaf_categ_id,
            session_start_dt=session_start_dt, exclude=0),
        Row(item_id=3, soj='dummy=dummy&viwtbidstr=937', site_id=site_id, leaf_categ_id=leaf_categ_id,
            session_start_dt=session_start_dt, exclude=0),
        Row(item_id=4, soj='viwtbidstr=937%2C939', site_id=site_id, leaf_categ_id=leaf_categ_id,
            session_start_dt=session_start_dt, exclude=0),  # filtered by limit
        Row(item_id=5, soj='viwtbidstr=937%2C939', site_id=site_id, leaf_categ_id=leaf_categ_id,
            session_start_dt=session_start_dt, exclude=0),
        Row(item_id=5, soj='viwtbidstr=937%2C939', site_id=site_id, leaf_categ_id=leaf_categ_id,
            session_start_dt=session_start_dt, exclude=0),
        Row(item_id=6, soj='soj1', site_id=site_id, leaf_categ_id=leaf_categ_id,
            session_start_dt='2025-03-04', exclude=0),  # date filtered out
        Row(item_id=7, soj='soj1', site_id=site_id, leaf_categ_id=leaf_categ_id,
            session_start_dt='2025-03-06', exclude=0),  # date filtered out
        Row(item_id=8, soj='soj1', site_id=site_id, leaf_categ_id=leaf_categ_id,
            session_start_dt=session_start_dt, exclude=1),  # filtered out by exclude
    ]
    expected_data = [
        Row(item_id=1, site_id=site_id, leaf_categ_id=leaf_categ_id, rank=VACANT_SLOT_RANK),
        Row(item_id=2, site_id=site_id, leaf_categ_id=leaf_categ_id, rank=VACANT_SLOT_RANK),
        Row(item_id=3, site_id=site_id, leaf_categ_id=leaf_categ_id, rank=VACANT_SLOT_RANK),
        Row(item_id=5, site_id=site_id, leaf_categ_id=leaf_categ_id, rank=2),
    ]
    vi_df = spark.createDataFrame(vi_data)
    vi_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.VI_EVENT_FACT")
    expected_df = spark.createDataFrame(expected_data)

    with mock.patch('generative_signals.etl.item_selection.LIMIT_ITEMS_FACTOR', 1):
        result_df = get_eligible_viewed_item_ids(spark, session_start_dt, session_start_dt, 4)

    assert_dataframe_equal(expected_df, result_df)


def test_filter_items_relevant_cats(spark):
    items_data = [
        Row(item_id=1, leaf_categ_id=101, SITE_ID=1, title='title1'),
        Row(item_id=2, leaf_categ_id=102, SITE_ID=1, title='title2'),
        Row(item_id=3, leaf_categ_id=103, SITE_ID=2, title='title3'),  # filtered out by site_id
        Row(item_id=3, leaf_categ_id=104, SITE_ID=1, title='title4')  # filtered out by leaf_categ_id
    ]
    cats_data = [
        Row(leaf_categ_id=101, SITE_ID=1),
        Row(leaf_categ_id=102, SITE_ID=1),
        Row(leaf_categ_id=103, SITE_ID=1),
    ]
    expected_data = [
        Row(item_id=1),
        Row(item_id=2)
    ]
    items_df = spark.createDataFrame(items_data)
    cats_df = spark.createDataFrame(cats_data)
    expected_df = spark.createDataFrame(expected_data)

    result_df = filter_items_relevant_cats(items_df, cats_df)

    assert_dataframe_equal(expected_df, result_df)


def test_filter_previously_processed_item_ids(spark):
    broadcast_items_ids_df = spark.createDataFrame([Row(item_id=1), Row(item_id=2), Row(item_id=3)])
    previously_processed_item_ids_df = spark.createDataFrame([Row(item_id=1), Row(item_id=3)])
    expected_df = spark.createDataFrame([Row(item_id=2)])

    with mock.patch('generative_signals.etl.item_selection.get_previously_processes_item_ids_df',
                    return_value=previously_processed_item_ids_df):
        result_df = filter_previously_processed_item_ids(spark, broadcast_items_ids_df, "2025-03-30", True)

    assert_dataframe_equal(expected_df, result_df)


def test_get_items_aspects(spark):
    session_start_dt = '2025-03-05'
    past_start_dt = '2025-03-04'
    futures_start_dt = '2025-03-06'
    spark.conf.set("spark.sql.mapKeyDedupPolicy", "LAST_WIN")

    valid_aspect_type = VALID_ASPECT_TYPES[0]
    invalid_src = INVALID_ASPECT_SRC[0]
    valid_src = 0
    invalid_value = INVALID_ASPECT_VALUES[0]
    items_aspects_data = [
        Row(item_id=1, auct_end_dt=session_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
            PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM='value1'),
        Row(item_id=1, auct_end_dt=session_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
            PRDCT_ASPCT_NM='aspect2', ASPCT_VLU_NM='value2'),
        Row(item_id=2, auct_end_dt=futures_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
            PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM='value-filtered'),  # first vlu_nm is overridden by LAST_WIN mapKeyDedupPolicy
        Row(item_id=2, auct_end_dt=futures_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
            PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM='value1'),
        Row(item_id=3, auct_end_dt=past_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
            PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM='value1'),  # filtered by date
        Row(item_id=4, auct_end_dt=session_start_dt, NS_TYPE_CD='invalid', ASPCT_SRC=valid_src,
            PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM='value1'),  # filtered by NS_TYPE_CD
        Row(item_id=5, auct_end_dt=session_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=invalid_src,
            PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM='value1'),  # filtered by ASPCT_SRC
        Row(item_id=6, auct_end_dt=session_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
            PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM=invalid_value),  # filtered by ASPCT_VLU_NM
        Row(item_id=7, auct_end_dt=session_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
            PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM=None),  # filtered by ASPCT_VLU_NM
        Row(item_id=8, auct_end_dt=session_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
            PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM=None),  # filtered by broadcast_new_items_ids_df
    ]
    expected_data = [
        Row(item_id=1, aspects={'aspect1': 'value1', 'aspect2': 'value2'}),
        Row(item_id=2, aspects={'aspect1': 'value1'}),
    ]
    items_aspects_df = spark.createDataFrame(items_aspects_data)
    items_aspects_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.ITEM_ASPCT_CLSSFCTN_SAP")
    broadcast_new_items_ids_df = spark.createDataFrame([Row(item_id=i) for i in range(8)])
    expected_df = spark.createDataFrame(expected_data)

    result_df = get_items_aspects(spark, broadcast_new_items_ids_df, session_start_dt)

    assert_dataframe_equal(expected_df, result_df)


def test_extract_items_for_generation__happy_flow(spark):
    session_start_dt = '2025-03-05'
    site_id = 0
    business_vertical_name = 'Fashion'
    filter_categ_lvl3_id = 176992
    vi_data = [
        Row(item_id=1, soj='dummy', site_id=site_id, leaf_categ_id=101, exclude=0,
            session_start_dt=session_start_dt),
        Row(item_id=2, soj='dummy', site_id=site_id, leaf_categ_id=102, exclude=0,
            session_start_dt=session_start_dt),
        Row(item_id=3, soj='dummy', site_id=site_id, leaf_categ_id=103, exclude=0,
            session_start_dt=session_start_dt),  # cat not in business_vertical_name
        Row(item_id=4, soj='dummy', site_id=site_id, leaf_categ_id=102, exclude=0,
            session_start_dt=session_start_dt),  # processed previously
        Row(item_id=5, soj='dummy', site_id=site_id, leaf_categ_id=104, exclude=0,
            session_start_dt=session_start_dt),  # filtered lvl3
    ]
    cats_data = [
        Row(leaf_categ_id=101, MOVE_TO=101, SITE_ID=site_id, bsns_vrtcl_name=business_vertical_name, categ_lvl3_id=1),
        Row(leaf_categ_id=102, MOVE_TO=102, SITE_ID=site_id, bsns_vrtcl_name=business_vertical_name, categ_lvl3_id=1),
        Row(leaf_categ_id=103, MOVE_TO=103, SITE_ID=site_id, bsns_vrtcl_name='other vertical', categ_lvl3_id=1),
        Row(leaf_categ_id=104, MOVE_TO=104, SITE_ID=site_id, bsns_vrtcl_name=business_vertical_name,
            categ_lvl3_id=filter_categ_lvl3_id),
    ]
    listing_data = [Row(item_id=i, AUCT_TITL=f'title{i}', LSTG_STATUS_ID=0) for i in range(len(vi_data))]
    valid_aspect_type = VALID_ASPECT_TYPES[0]
    valid_src = 0
    desc_data = [Row(item_id=i, item_desc=f'item_desc{i}') for i in range(len(vi_data))]
    aspects_data = [Row(item_id=1, auct_end_dt=session_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
                        PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM='value1'),
                    Row(item_id=1, auct_end_dt=session_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
                        PRDCT_ASPCT_NM='aspect2', ASPCT_VLU_NM='value2'),
                    Row(item_id=2, auct_end_dt=session_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
                        PRDCT_ASPCT_NM='aspect1', ASPCT_VLU_NM='value1'),]
    expected_data = [
        Row(item_id=1, desc='item_desc1', aspects={'aspect1': 'value1', 'aspect2': 'value2'}, title='title1'),
        Row(item_id=2, desc='item_desc2', aspects={'aspect1': 'value1'}, title='title2')
    ]
    vi_df = spark.createDataFrame(vi_data)
    vi_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.VI_EVENT_FACT")
    cats_df = spark.createDataFrame(cats_data)
    cats_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.DW_CATEGORY_GROUPINGS")
    listing_df = spark.createDataFrame(listing_data)
    listing_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.DW_LSTG_ITEM")
    desc_df = spark.createDataFrame(desc_data)
    desc_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.LSTG_ITEM_DESC_DIM")
    aspects_df = spark.createDataFrame(aspects_data)
    aspects_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.ITEM_ASPCT_CLSSFCTN_SAP")
    previously_processed_item_ids_df = spark.createDataFrame([Row(item_id=4)])
    expected_df = spark.createDataFrame(expected_data)

    business_vertical_names = [business_vertical_name]
    excluded_categ_lvl3_id = [filter_categ_lvl3_id]
    site_ids = [site_id]
    with (mock.patch('generative_signals.etl.item_selection.get_previously_processes_item_ids_df',
                     return_value=previously_processed_item_ids_df),
          mock.patch('generative_signals.etl.item_selection.validate_output_path_is_empty',
                     return_value=True),
          mock.patch('generative_signals.etl.item_selection.persist_items_to_previously_processed',
                     return_value=True)):
        result_df = extract_items_for_generation(spark, session_start_dt, session_start_dt, "", True,
                                                 business_vertical_names=business_vertical_names, site_ids=site_ids,
                                                 excluded_categ_lvl3_id=excluded_categ_lvl3_id)

    assert_dataframe_equal(expected_df, result_df)


def test_extract_items_for_generation_with_non_vacant__happy_flow(spark):
    session_start_dt = '2025-03-05'
    site_id = 0
    business_vertical_name = 'Fashion'
    filter_categ_lvl3_id = 176992
    vi_data = [
        Row(item_id=1, soj='dummy', site_id=site_id, leaf_categ_id=101, exclude=0,
            session_start_dt=session_start_dt),
        Row(item_id=2, soj='dummy', site_id=site_id, leaf_categ_id=101, exclude=0,
            session_start_dt=session_start_dt),
        Row(item_id=3, soj='viwtbidstr=937%2C939', site_id=site_id, leaf_categ_id=101, exclude=0,
            session_start_dt=session_start_dt),  # filtered by limit
        Row(item_id=4, soj='viwtbidstr=937%2C939', site_id=site_id, leaf_categ_id=101, exclude=0,
            session_start_dt=session_start_dt),
        Row(item_id=4, soj='viwtbidstr=937%2C939', site_id=site_id, leaf_categ_id=101, exclude=0,
            session_start_dt=session_start_dt),
        Row(item_id=5, soj='viwtbidstr=937%2C939', site_id=site_id, leaf_categ_id=101, exclude=0,
            session_start_dt=session_start_dt),
        Row(item_id=5, soj='viwtbidstr=937%2C939', site_id=site_id, leaf_categ_id=101, exclude=0,
            session_start_dt=session_start_dt),
    ]
    cats_data = [
        Row(leaf_categ_id=101, MOVE_TO=101, SITE_ID=site_id, bsns_vrtcl_name=business_vertical_name, categ_lvl3_id=1),
    ]
    listing_data = [Row(item_id=i, AUCT_TITL=f'title{i}', LSTG_STATUS_ID=0) for i in range(len(vi_data))]
    valid_aspect_type = VALID_ASPECT_TYPES[0]
    valid_src = 0
    desc_data = [Row(item_id=i, item_desc=f'item_desc{i}') for i in range(len(vi_data))]
    aspects_data = [Row(item_id=i, auct_end_dt=session_start_dt, NS_TYPE_CD=valid_aspect_type, ASPCT_SRC=valid_src,
                        PRDCT_ASPCT_NM=f'aspect{i}', ASPCT_VLU_NM=f'value{i}') for i in range(len(vi_data))]
    expected_data = [Row(item_id=i, desc=f'item_desc{i}', aspects={f'aspect{i}': f'value{i}'}, title=f'title{i}')
                     for i in [1, 2, 4, 5]]
    vi_df = spark.createDataFrame(vi_data)
    vi_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.VI_EVENT_FACT")
    cats_df = spark.createDataFrame(cats_data)
    cats_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.DW_CATEGORY_GROUPINGS")
    listing_df = spark.createDataFrame(listing_data)
    listing_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.DW_LSTG_ITEM")
    desc_df = spark.createDataFrame(desc_data)
    desc_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.LSTG_ITEM_DESC_DIM")
    aspects_df = spark.createDataFrame(aspects_data)
    aspects_df.write.mode("overwrite").saveAsTable(name="ACCESS_VIEWS.ITEM_ASPCT_CLSSFCTN_SAP")
    previously_processed_item_ids_df = spark.createDataFrame([Row(item_id=-1)])
    expected_df = spark.createDataFrame(expected_data)

    business_vertical_names = [business_vertical_name]
    excluded_categ_lvl3_id = [filter_categ_lvl3_id]
    site_ids = [site_id]
    with (mock.patch('generative_signals.etl.item_selection.get_previously_processes_item_ids_df',
                     return_value=previously_processed_item_ids_df),
          mock.patch('generative_signals.etl.item_selection.validate_output_path_is_empty',
                     return_value=True),
          mock.patch('generative_signals.etl.item_selection.persist_items_to_previously_processed',
                     return_value=True)):
        result_df = extract_items_for_generation(spark, session_start_dt, session_start_dt, "", True,
                                                 business_vertical_names=business_vertical_names, site_ids=site_ids,
                                                 excluded_categ_lvl3_id=excluded_categ_lvl3_id, limit_items=4)

    assert_dataframe_equal(expected_df, result_df)
