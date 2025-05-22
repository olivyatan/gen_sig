# from pyspark.sql import SparkSession
from pyspark.sql.types import MapType, StringType
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import broadcast
import warnings

from dev.data_extraction.extraction_funcs_utils import filter_aspects
from dev.data_extraction.extraction_funcs_utils import invalid_aspect_names, invalid_aspct_src, forbidden_words,forbidden_words_post

warnings.filterwarnings("ignore")
import base64
from bs4 import BeautifulSoup
import re
from nltk.tokenize import sent_tokenize


valid_types = ['df']
invalid_values = ["", "null",  "none", "na", "n/a", "no", "unknown","false", "unbranded", "other"]

def html_to_plain(html: str):
    if html is None or len(html) == 0:
        return ""
    html = base64.b64decode(html).decode('utf-8')
    try:
        bs = BeautifulSoup(html, 'html.parser')
        plain = bs.get_text(separator=' ', strip=True)
        return re.sub(r"\s+", " ", plain)
    except:
        return ""

@F.udf(StringType())
def html_format_to_plain_udf(html):
    """
    Decodes Base64-encoded HTML and extracts readable plain text.

    Parameters:
    html (str): A Base64-encoded HTML string.

    Returns:
    str: Cleaned plain text.
    """
    if html is None or len(html) == 0:
        return ""

    try:
        # Decode Base64
        decoded_html = base64.b64decode(html).decode('utf-8')

        # Parse HTML to extract text
        soup = BeautifulSoup(decoded_html, 'html.parser')
        plain_text = soup.get_text(separator=' ', strip=True)

        # Clean excess whitespace
        return re.sub(r"\s+", " ", plain_text).strip()

    except Exception as e:
        return f"Error decoding text: {e}"

def lowercase_string_columns(df):
    string_cols = [item[0] for item in df.dtypes if item[1].startswith('string') and item[0] != 'GALLERY_URL']
    for col_name in string_cols:
        df = df.withColumn(col_name, F.lower(F.col(col_name)))
    return df


combineMap = F.udf(lambda maps: {key: f[key] for f in maps for key in f},
                   MapType(StringType(), StringType()))


def split_to_sentences(text):
    # Split the text into sentences
    sentences = sent_tokenize(text)
    return sentences


def remove_forbidden_tokens_from_dict(aspects_dict, forbidden_tokens = forbidden_words):
    if aspects_dict is None or len(aspects_dict) == 0:
        return {}
    res_dict = {}
    for k,v in aspects_dict.items():
        k_f, v_f = remove_forbidden_tokens(k).strip(), remove_forbidden_tokens(v).strip()
        if len(k_f) > 0 and len(v_f) > 0:
            res_dict[k_f] = v_f
    return res_dict



def remove_forbidden_tokens(text, forbidden_tokens = forbidden_words):
    if text is None or len(text) == 0:
        return ""
    sentences = split_to_sentences(text.lower())
    # Create a regex pattern to match any of the forbidden tokens
    pattern = re.compile('|'.join(map(re.escape, forbidden_tokens)), re.IGNORECASE)

    # Filter out sentences that contain any of the forbidden tokens
    filtered_sentences = [sentence for sentence in sentences if not pattern.search(sentence)]
    return ' '.join(filtered_sentences)


#Ad-hoc cleanups. Should be integrated in the spark extraction script.
def post_extraction_amendments(data_pdf, should_remove_forbidden_tokens=True, apply_html_to_plain = False):

    data_pdf.rename(columns={'ITEM_ID': 'item_id', 'ASPECTS': 'aspects', 'AUCT_TITL': 'title'}, inplace=True)
    if apply_html_to_plain:
        data_pdf['desc'] = data_pdf['item_desc'].apply(html_to_plain)
    data_pdf['aspects'] = data_pdf['aspects'].apply(filter_aspects)
    if should_remove_forbidden_tokens:
        data_pdf['desc_no_forbidden'] = data_pdf['desc'].apply(lambda x : remove_forbidden_tokens(x))
        data_pdf['title_no_forbidden'] = data_pdf['title'].apply(lambda x : remove_forbidden_tokens(x))
        data_pdf['aspects_no_forbidden'] = data_pdf['aspects'].apply(lambda x : remove_forbidden_tokens_from_dict(x))

    return data_pdf


def collect_item_aspects(session, df_items, outcols, df_aspects, take_shortest_dup_value=True):
    df_items = df_items.dropDuplicates(['ITEM_ID', "AUCT_END_DT"])

    # max_auct_end_dt = df_items.agg({"AUCT_END_DT": "max"}).collect()[0][0]
    # min_auct_end_dt = df_items.agg({"AUCT_END_DT": "min"}).collect()[0][0]

    # get aspects
    df_item_aspects = df_aspects.join(broadcast(df_items), on=['ITEM_ID', "AUCT_END_DT"])

    df_item_aspects = df_item_aspects.filter(
        (F.col("ASPCT_VLU_NM").isNotNull()) &
        (~F.lower(F.col("ASPCT_VLU_NM")).isin(invalid_values)))

    # in case of multiple values per aspect, select the one with the shortest string value
    if take_shortest_dup_value:
        df_item_aspects = df_item_aspects.withColumn('VALUE_LEN', F.length("ASPCT_VLU_NM").alias('VALUE_LEN'))
        row_column = F.row_number().over(Window.partitionBy("ITEM_ID", "PRDCT_ASPCT_NM")
                                         .orderBy(df_item_aspects['VALUE_LEN']))
        df_item_aspects = df_item_aspects.withColumn("row_num", row_column.alias("row_num")) \
            .where(F.col("row_num") == 1)
    else:
        df_item_aspects = df_item_aspects.groupBy(*(outcols + ["PRDCT_ASPCT_NM"])) \
            .agg(F.first("ASPCT_VLU_NM").alias("ASPCT_VLU_NM"))

    df_item_aspects = lowercase_string_columns(df_item_aspects)
    # This filtering is done before this function invocation, so as to avoid computation over invalid rows
    # df_item_aspects = df_item_aspects.where((F.col("NS_TYPE_CD").isin(valid_types))
    #                                         & (~F.col("PRDCT_ASPCT_NM").isin(invalid_aspect_names)))

    # group all aspect name-value rows per item into a row with a dict column
    df_item_aspects = df_item_aspects \
        .withColumn("ASPECTS", F.create_map("PRDCT_ASPCT_NM", "ASPCT_VLU_NM").alias("ASPECTS")) \
        .groupBy(*outcols) \
        .agg(F.collect_list('ASPECTS').alias('ASPECTS')) \
        .select(*(outcols + [combineMap('ASPECTS').alias('ASPECTS')]))
    return df_item_aspects


# !!TODO: check whether we want auction start date or auction end date, or maybe alive status instead.
def extract_item_data(spark, start_date, end_date, metacats=None, business_vertical_names=None,
                      leafcats=None, site_ids=None, n_items=None,
                      take_shortest_dup_value=True, excluded_leafs=None, excluded_categ_lvl2_id=None,
                      excluded_categ_lvl3_id=None):
    items_df = spark.table("ACCESS_VIEWS.DW_LSTG_ITEM").filter((F.col('auct_end_dt') >= start_date) & \
                                                               (F.col('auct_end_dt') <= end_date)).coalesce(10000)

    cats_table = spark.table('ACCESS_VIEWS.DW_CATEGORY_GROUPINGS') \
        .withColumnRenamed('leaf_categ_id', 'leaf_categ_id_cat_table') \
        .filter(F.col('leaf_categ_id_cat_table') == F.col('MOVE_TO'))

    if leafcats is not None:
        items_df = items_df.filter(F.col('leaf_categ_id').isin(leafcats))
        cats_table = cats_table.filter(F.col('leaf_categ_id_cat_table').isin(leafcats))

    if site_ids is not None:
        items_df = items_df.filter(F.col('ITEM_SITE_ID').isin(site_ids))
        cats_table = cats_table.filter(F.col('site_id').isin(site_ids))

    # Exclude specific leaf categories if provided
    if excluded_leafs is not None:
        items_df = items_df.filter(~F.col('leaf_categ_id').isin(excluded_leafs))
        cats_table = cats_table.filter(~F.col('leaf_categ_id_cat_table').isin(excluded_leafs))

    if excluded_categ_lvl2_id is not None:
        cats_table = cats_table.filter(~F.col('categ_lvl2_id').isin(excluded_categ_lvl2_id))

    # Exclude specific Level 3 categories if provided
    if excluded_categ_lvl3_id is not None:
        cats_table = cats_table.filter(~F.col('categ_lvl3_id').isin(excluded_categ_lvl3_id))

    if (business_vertical_names is not None) or (metacats is not None):
        if metacats is not None:
            cats_table = cats_table.filter(F.col('meta_categ_id').isin(metacats))
        if business_vertical_names is not None:
            # prod_ref_df = prod_ref_df.filter(F.col('DMNT_VRTCL_NAME').isin(business_vertical_names))
            cats_table = cats_table.filter(F.col('bsns_vrtcl_name').isin(business_vertical_names))

        items_df = items_df.join(
            broadcast(cats_table),
            (items_df['leaf_categ_id'] == cats_table['leaf_categ_id_cat_table']) &
            (items_df['ITEM_SITE_ID'] == cats_table['SITE_ID'])
            , how='inner'
        )
    else:
        items_df = items_df.join(
            broadcast(cats_table),
            (items_df['leaf_categ_id'] == cats_table['leaf_categ_id_cat_table']) &
            (items_df['ITEM_SITE_ID'] == cats_table['SITE_ID'])
            , how='left'
        )

    if n_items is not None:
        # select n_items random items
        items_df = items_df.sample(False, 0.01).limit(n_items)

    outcols = ['ITEM_ID', 'AUCT_TITL', 'SUBTITLE', 'AUCT_START_DT', 'AUCT_END_DT', 'leaf_categ_id', \
               'ITEM_SITE_ID', 'bsns_vrtcl_name', 'meta_categ_id', 'categ_lvl3_id', 'GALLERY_URL']
    items_df = items_df.select(*outcols)  # .cache()

    description_df = spark.table("ACCESS_VIEWS.LSTG_ITEM_DESC_DIM").filter((F.col('auct_end_dt') >= start_date) & \
                                                                           (F.col('auct_end_dt') <= end_date)) \
        .select('item_id', 'item_desc')
    aspects_df = spark.table("ACCESS_VIEWS.ITEM_ASPCT_CLSSFCTN_SAP").filter(
        (F.col('auct_end_dt') >= start_date) &
        (F.col('auct_end_dt') <= end_date) &
        (F.col("NS_TYPE_CD").isin(valid_types)) &
        (~F.lower(F.col("PRDCT_ASPCT_NM")).isin(invalid_aspect_names)) &
        (~F.col("ASPCT_SRC").isin(invalid_aspct_src)))

    print("Adding aspects")
    df_items_aspects = collect_item_aspects(spark, items_df, outcols, aspects_df,
                                            take_shortest_dup_value=take_shortest_dup_value
                                            )
    print("Adding descriptions")
    df_item_aspects_descr = description_df.join(broadcast(df_items_aspects), on='item_id', how='right')

    # df_item_aspects_descr = df_item_aspects_descr.withColumn("desc", html_format_to_plain_udf(F.col('item_desc')))

    # items_df.unpersist()
    df_item_aspects_descr = df_item_aspects_descr.withColumnRenamed('ITEM_ID', 'item_id') \
        .withColumnRenamed('ASPECTS', 'aspects') \
        .withColumnRenamed('AUCT_TITL', 'title')

    # df_item_aspects_descr['desc'] = df_item_aspects_descr['item_desc'].apply(html_to_plain)
    return df_item_aspects_descr













