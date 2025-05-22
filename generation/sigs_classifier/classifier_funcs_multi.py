import json
import re
from pychomsky.chchat import EbayLLMChatWrapper
from langchain.schema.messages import HumanMessage
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs
import os
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from ellement.transformers import AutoModelForCausalLM, AutoTokenizer
from ellement.peft import PeftConfig, PeftModel
from sklearn.model_selection import KFold
import tiktoken

# Obtain the Hadoop classpath
import socket
hostname = socket.gethostname()

if hostname[0].startswith('krylov'):
    on_kry = True
else:
    on_kry = False
if on_kry:
    classpath = subprocess.run(['hadoop', 'classpath', '--glob'], capture_output=True)
    os.environ["CLASSPATH"] = classpath.stdout.decode()
    os.environ['LD_LIBRARY_PATH'] = os.getenv('HADOOP_HOME') + '/lib/native/:' + os.getenv('LD_LIBRARY_PATH')

USER_CONTENT_BASIC_EXPLANATION = f'''
You are an expert judge. You'll now be presented with an eBay's listing data,\
consisting of the listing title, aspects and description, followed by an automatically generated characteristics of\
this listing, which is a candidate to be shown to eBay shoppers as an incentivising signal when viewing the listing.\
Your job is to indicate whether this characteristic is a good signal, in the sense that it's safe, informative\
incentivising for purchase, and is not trivial (i.e. the information is not obvious from the title or probably the\
image). Please return label 1 for a good signal, label 2 for an excellent signal, and label 0 to\ 
indicate it's not a valid signal. Also optionally add a reason and/or comment on the chosen label. Here is the\
listing data:
\n Item title: {{0}}, \n Item aspects: {{1}}, \n Item description: {{2}}. \n Generated signal: {{3}}
'''
USER_CONTENT_EXPLANATION_ORTAL = f'''
You are an expert judge. You'll now be presented with an eBay's product data, consisting of the product title, aspects\
and description, followed by an automatically generated characteristics of this product, which is a candidate to be\
shown to eBay shoppers as an incentivising signal when viewing the listing. Based on the product's details and the \
provided signal, assign a label indicating whether the signal is good. Provide a reason for your decision. Here are the\
possible labels and reasons to assign them:

Label 0: Bad Signal. Signal is irrelevant, unclear, doesn't provide new information over the product's title,
or dominated by extraneous elements (e.g., excessive numbers or technical jargon).
Possible Reasons for Label 0:
Use the following reasons to justify why the signal is bad.
-Repetitive Tokens: Contains redundant words or phrases.
-Not Clear: Vague or lacks meaningful information.
-Doesn't add information to the title: signal appears partially in title
-Refers to the product's condition or return policy.
-Describes the Store or seller.
-Obvious from image. 
-Too Technical: Focuses only on specs without linking to user benefits.
-Incorrect Grammar.
-Excessive Length: Overly verbose, detracting from clarity.
-Off-Topic: Does not provide relevant or valuable product information.
-Problematic Numerical Pattern: Contains numerical patterns that are misleading, confusing, or not contextualized properly (e.g., random numbers or irrelevant measurements. 
-Numerical Dominant: Overloaded with numbers, making it difficult to extract meaningful product information.

*The reason should clearly explain why the signal does not meet the criteria for a good or urgent signal.

1: Good Signal. The signal is clear, relevant, and provides useful information about the product, which is not inculded in the product's title.
2: Very Good Signal. In addition to being a good signal, it incentivizes the user to purchase by highlighting unique, desirable, or urgent features. 


- Examples: 
Label 0 (Bad Signal):The signal is irrelevant, unclear, exhibits numerical dominance or repeats information in the title. examples: 

  - 40 Pieces Per Pack
  - Four Sizes And Options
  - Compatible Part Number
  - USB Or Battery Powered
  - 2-Pc. 5-In. Gnomes
  - Genuine Samsung Da29-00003G
  - 18V Lithium 2Ah Battery
  - Fits 26 - 50 Lbs
  - 3% Spandex
  - Dry Wash Only
  - Easy Returns 

  Label 1 (Good Signal): The signal is clear, relevant, and provides useful information about the product, which is not predent in the product's title.
  Examples:
  - 97% Natural Ingredients
  - Smooth Rotating Design
  - Durable Stainless Steel
  - Multi-Sizing Options
  - USDA Organic Certified
  - Non-Stick Surface
  - Strong Wind Speed 
  - Playful Print

  Label 2 (Very Good Signal):** In addition to being a good signal (1) the signal incentivizes the user to purchase.
  Examples:
  - Natural Solid Wood Material
  - 5 Years Manufacturer Warranty
  - Solar Powered
  - Lightweight Design
  
Here is the listing data:
\n Item title: {{0}}, \n Item aspects: {{1}}, \n Item description: {{2}}. \n Generated signal: {{3}}
'''


USER_CONTENT_EXPLANATION_ORTAL_MULTI = f'''
You are an expert judge. You'll now be presented with an eBay's product data, consisting of the product title, aspects\
and description, followed by an automatically generated characteristics of this product, which is a candidate to be\
shown to eBay shoppers as an incentivising signal when viewing the listing. Based on the product's details and the \
provided signal, assign a label indicating whether the signal is good. Provide a reason for your decision. Here are the\
possible labels and reasons to assign them:

Label 0: Bad Signal. Signal is irrelevant, unclear, doesn't provide new information over the product's title,
or dominated by extraneous elements (e.g., excessive numbers or technical jargon).
Possible Reasons for Label 0:
Use the following reasons to justify why the signal is bad.
-Repetitive Tokens: Contains redundant words or phrases.
-Not Clear: Vague or lacks meaningful information.
-Doesn't add information to the title: signal appears partially in title
-Refers to the product's condition or return policy.
-Describes the Store or seller.
-Obvious from image. 
-Too Technical: Focuses only on specs without linking to user benefits.
-Incorrect Grammar.
-Excessive Length: Overly verbose, detracting from clarity.
-Off-Topic: Does not provide relevant or valuable product information.
-Problematic Numerical Pattern: Contains numerical patterns that are misleading, confusing, or not contextualized properly.
-Numerical Dominant: Overloaded with numbers, making it difficult to extract meaningful product information.

*The reason should clearly explain why the signal does not meet the criteria for a good or urgent signal.

1: Good Signal. The signal is clear, relevant, and provides useful information about the product, which is not inculded in the product's title.
2: Very Good Signal. In addition to being a good signal, it incentivizes the user to purchase by highlighting unique, desirable, or urgent features. 

- Examples: 
Label 0 (Bad Signal):The signal is irrelevant, unclear, exhibits numerical dominance or repeats information in the title. examples: 

  - 40 Pieces Per Pack
  - Four Sizes And Options
  - Compatible Part Number
  - USB Or Battery Powered
  - 2-Pc. 5-In. Gnomes
  - Genuine Samsung Da29-00003G
  - 18V Lithium 2Ah Battery
  - 3% Spandex
  - Dry Wash Only
  - Easy Returns 

  Label 1 (Good Signal): The signal is clear, relevant, and provides useful information about the product, which is not present in the product's title.
  Examples:
  - 97% Natural Ingredients
  - Smooth Rotating Design
  - Durable Stainless Steel
  - Multi-Sizing Options
  - USDA Organic Certified
  - Non-Stick Surface
  - Strong Wind Speed 
  - Playful Print

  Label 2 (Very Good Signal):** In addition to being a good signal (1) the signal incentivizes the user to purchase.
  Examples:
  - Natural Solid Wood Material
  - 5 Years Manufacturer Warranty
  - Solar Powered
  - Lightweight Design
  - 95% Silk
  - Ultra-Soft And Super Absorbent

Here is the listing data:
\n Item title: {{0}}, \n Item aspects: {{1}}, \n Item description: {{2}}. \n Generated signal: {{3}} 

Please return the response in this format **only** :
Label: <one of: 0, 1, or 2>  
Reason: <one reason from the list provided above> 
'''





ASSISTANT_ONLY_LABEL_CONTENT = f"Label: {{0}}"
ASSISTANT_CONTENT_LABEL_AND_REASON = assistant_content = f"Label: {{0}}, Reason: {{1}}"
ASSISTANT_MULTI_LABEL_RESPONSE = """Labels: {0}\nReasons: {1}"""

JUDGE_PROMPT_COLNAME = 'judge_prompt'

ASPECTS_COLNAME = 'aspects'
TITLE_COLNAME = 'title'
DESCRIPTION_COLNAME = 'desc'
GENERATED_SIG_COLNAME = 'gen_sig_gpt'
LABEL_COLNAME = 'final_label'
REASON_COLNAME = 'final_reason'
COMMENT_COLNAME = 'final_comment'
MANUAL_REASONS_COLNAME = 'reason_manual'


class PromptFormattedStr():
    def __init__(self, content_format_str, fields_names_to_extract):
        self.content_format_str = content_format_str
        self.fields_names_to_extract = fields_names_to_extract

    def format(self, row, how='numerical_label'):
        if how is None:
            format_list = [str(row.get(field)).strip() for field in self.fields_names_to_extract]
        elif how == 'numerical_label':
            format_list = [
                str(row.get(field)).strip() if field != LABEL_COLNAME else int(float(row.get(field)))
                for field in self.fields_names_to_extract
            ]
        elif how == 'numerical_binary_label':
            format_list = [
                str(row.get(field)).strip() if field != LABEL_COLNAME else (1 if int(float(row.get(field))) != 0 else 0)
                for field in self.fields_names_to_extract
            ]
        else:
            raise ValueError(f"`how` should be one of ['numerical_label', 'numerical_binary_label', None]")

        return self.content_format_str.format(*format_list)


prompts = {
    # === SINGLE SIGNAL PROMPTS ===
    'assistant_only_label': {
        'user_content': PromptFormattedStr(
            USER_CONTENT_BASIC_EXPLANATION,
            [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, GENERATED_SIG_COLNAME]
        ),
        'assistant_content': PromptFormattedStr(
            ASSISTANT_ONLY_LABEL_CONTENT,
            [LABEL_COLNAME]
        ),
        'is_multi': False
    },

    'assistant_only_label_ortal_prompt': {
        'user_content': PromptFormattedStr(
            USER_CONTENT_EXPLANATION_ORTAL,
            [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, GENERATED_SIG_COLNAME]
        ),
        'assistant_content': PromptFormattedStr(
            ASSISTANT_ONLY_LABEL_CONTENT,
            [LABEL_COLNAME]
        ),
        'is_multi': False
    },

    'assistant_label_and_explanation': {
        'user_content': PromptFormattedStr(
            USER_CONTENT_BASIC_EXPLANATION,
            [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, GENERATED_SIG_COLNAME]
        ),
        'assistant_content': PromptFormattedStr(
            ASSISTANT_ONLY_LABEL_CONTENT,
            [LABEL_COLNAME, REASON_COLNAME + COMMENT_COLNAME]
        ),
        'is_multi': False
    },

    'assistant_label_and_single_reason_ortal_prompt': {
        'user_content': PromptFormattedStr(
            USER_CONTENT_EXPLANATION_ORTAL,
            [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, GENERATED_SIG_COLNAME]
        ),
        'assistant_content': PromptFormattedStr(
            ASSISTANT_CONTENT_LABEL_AND_REASON,
            [LABEL_COLNAME, MANUAL_REASONS_COLNAME]
        ),
        'is_multi': False
    },

    'assistant_label_and_reason_multi_signal_ortal_prompt': {
        'user_content': PromptFormattedStr(
            USER_CONTENT_EXPLANATION_ORTAL_MULTI,
            [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, 'grouped_signals']
        ),
        'assistant_content': PromptFormattedStr(
            ASSISTANT_MULTI_LABEL_RESPONSE,
            ['grouped_labels', 'grouped_reasons']
        ),
        'is_multi': True
    },
    'assistant_label_and_reason_multi_signal_ortal_prompt_1sig': {
        'user_content': PromptFormattedStr(
            USER_CONTENT_EXPLANATION_ORTAL_MULTI,
            [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, GENERATED_SIG_COLNAME]
        ),
        'assistant_content': PromptFormattedStr(
            ASSISTANT_CONTENT_LABEL_AND_REASON,
            [LABEL_COLNAME, MANUAL_REASONS_COLNAME]
        ),
        'is_multi': False
    },

    'assistant_label_ortal_prompt_nr_1sig': {
        'user_content': PromptFormattedStr(
            USER_CONTENT_EXPLANATION_ORTAL,
            [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, GENERATED_SIG_COLNAME]
        ),
        'assistant_content': PromptFormattedStr(
            ASSISTANT_CONTENT_LABEL_AND_REASON,
            [LABEL_COLNAME, MANUAL_REASONS_COLNAME]
        ),
        'is_multi': False
    }

}

#------- Preprocessing: make sure desc is up to 1000 tokens---
def truncate_text_to_tokens(text, keep_tokens=800):
    """
    Truncates the input text to the first `keep_tokens` tokens using GPT's tokenization.

    Parameters:
        text (str): Input text.
        keep_tokens (int): Number of tokens to retain from the start. Default is 800.

    Returns:
        str: Truncated text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # same as GPT-4 and GPT-3.5-turbo
    if pd.isna(text) or not isinstance(text, str):
        return text

    tokens = encoding.encode(text)
    truncated_tokens = tokens[:keep_tokens]
    return encoding.decode(truncated_tokens)



def write_to_hdfs(messages_df, out_filename, out_dir='/user/b_perso/mmandelbrod/gen_sigs/'):
    # Connect to HDFS using PyArrow
    hdfs = pyarrow.fs.HadoopFileSystem(host="default")  # Adjust 'host' as needed
    # Convert the pandas DataFrame to a pyarrow Table
    table = pa.Table.from_pandas(messages_df)
    outpath = os.path.join(out_dir, out_filename)
    # Write the pyarrow Table to HDFS as a Parquet file
    with hdfs.open_output_stream(outpath) as f:
        pq.write_table(table, f)
        print(f"Written to {outpath}")


def create_athena_dataset_one_sig_per_item(annotated_df, how='assistant_only_label'):
    if how not in prompts.keys():
        raise ValueError(f"how should be one of {prompts.keys()}")
    json_list = []
    user_content, assitant_content = prompts[how]['user_content'], prompts[how]['assistant_content']
    for index, row in annotated_df.iterrows():
        # print(f"{index}")
        row_list = [
            {
                "role": "user",
                "content": user_content.format(row)
            },
            {
                "role": "assistant",
                "content": assitant_content.format(row)
            }
        ]
        json_list.append(row_list)
    slst = [json.dumps(x) for x in json_list]
    df_messages = pd.DataFrame({"messages": slst})
    return df_messages

# Train set
def create_athena_dataset_multi(annotated_df, how):
    if how not in prompts:
        raise ValueError(f"how should be one of {prompts.keys()}")

    prompt_def = prompts[how]
    user_content, assistant_content = prompt_def['user_content'], prompt_def['assistant_content']
    is_multi = prompt_def.get('is_multi', False)

    json_list = []
    if is_multi:
        for item_id, group in annotated_df.groupby('item_id'):
            base_row = group.iloc[0]
            signals = '\n'.join([f"{idx + 1}. {sig}" for idx, sig in enumerate(group[GENERATED_SIG_COLNAME])])
            labels = ', '.join([str(int(l)) for l in group[LABEL_COLNAME]])
            reasons = ', '.join([str(r) for r in group[MANUAL_REASONS_COLNAME]])

            row_dict = base_row.to_dict()
            row_dict['grouped_signals'] = signals
            row_dict['grouped_labels'] = labels
            row_dict['grouped_reasons'] = reasons
            row_dict['num_signals'] = len(group)

            row_list = [
                {"role": "user", "content": user_content.format(row_dict)},
                {"role": "assistant", "content": assistant_content.format(row_dict)}
            ]
            json_list.append(row_list)
    else:
        for _, row in annotated_df.iterrows():
            row_list = [
                {"role": "user", "content": user_content.format(row)},
                {"role": "assistant", "content": assistant_content.format(row)}
            ]
            json_list.append(row_list)

    slst = [json.dumps(x) for x in json_list]
    return pd.DataFrame({"messages": slst})


def create_inference_dataset_one_sig_per_item(df_test, how):
    if how not in prompts.keys():
        raise ValueError(f"how should be one of {prompts.keys()}")
    user_content = prompts[how]['user_content']
    prompt_list = []

    for index, row in df_test.iterrows():
        curr_prompt = user_content.format(row)
        prompt_list.append(curr_prompt)
    df_test[JUDGE_PROMPT_COLNAME] = prompt_list
    return df_test

# Test set:
def create_inference_dataset_multi(df_test, how):
    if how not in prompts:
        raise ValueError(f"'how' should be one of {prompts.keys()}")

    prompt_def = prompts[how]
    user_content = prompt_def['user_content']
    is_multi = prompt_def.get('is_multi', False)

    rows = []

    if is_multi:
        for item_id, group in df_test.groupby('item_id'):
            base_row = group.iloc[0]
            sig_list = group[GENERATED_SIG_COLNAME].tolist()

            if len(sig_list) == 1:
                signals = sig_list[0]
            else:
                signals = '\n'.join([f"{idx + 1}. {sig}" for idx, sig in enumerate(sig_list)])

            row_dict = base_row.to_dict()
            row_dict['grouped_signals'] = signals
            row_dict['num_signals'] = len(sig_list)

            prompt = user_content.format(row_dict)

            rows.append({
                'item_id': item_id,
                'grouped_signals': signals,
                JUDGE_PROMPT_COLNAME: prompt
            })

        return pd.DataFrame(rows)

    else:
        # Handle the non-multi version by creating one row per signal
        prompt_list = []
        for _, row in df_test.iterrows():
            prompt = user_content.format(row)
            prompt_list.append({
                'item_id': row['item_id'],
                GENERATED_SIG_COLNAME: row[GENERATED_SIG_COLNAME],
                JUDGE_PROMPT_COLNAME: prompt
            })

        return pd.DataFrame(prompt_list)


def create_train_test_datasets_one_sig_per_item(annotated_df, train_size=0.7, how='assistant_only_label',
                                                hdfs_out_dirname=None,
                                                hdfs_out_dir_base='/user/b_perso/mmandelbrod/gen_sigs/',
                                                override=False
                                                ):
    annotated_df = annotated_df[~annotated_df[LABEL_COLNAME].isna()]
    df_train, df_test = train_test_split(annotated_df, train_size=0.7)
    df_messages_train = create_athena_dataset_one_sig_per_item(df_train, how=how)
    df_messages_test = create_inference_dataset_one_sig_per_item(df_test, how=how)
    if hdfs_out_dirname is not None:
        hdfs_write_path = os.path.join(hdfs_out_dir_base, hdfs_out_dirname)
        write_to_hdfs(df_messages_train, out_filename='train_athena.parquet', out_dir=hdfs_write_path)
        write_to_hdfs(df_messages_test, out_filename='test.parquet', out_dir=hdfs_write_path)
    return df_messages_train, df_messages_test


def create_inference_dataset_multi(df_test, how):
    if how not in prompts:
        raise ValueError(f"'how' should be one of {prompts.keys()}")

    prompt_def = prompts[how]
    user_content = prompt_def['user_content']
    is_multi = prompt_def.get('is_multi', False)

    rows = []

    if is_multi:
        for item_id, group in df_test.groupby('item_id'):
            base_row = group.iloc[0]
            sig_list = group[GENERATED_SIG_COLNAME].tolist()

            signals = '\n'.join([f"{idx + 1}. {sig}" for idx, sig in enumerate(sig_list)])

            row_dict = base_row.to_dict()
            row_dict['grouped_signals'] = signals

            prompt = user_content.format(row_dict)

            rows.append({
                'item_id': item_id,
                'grouped_signals': signals,
                JUDGE_PROMPT_COLNAME: prompt
            })

        return pd.DataFrame(rows)

    else:
        # Handle the non-multi version by creating one row per signal
        prompt_list = []
        for _, row in df_test.iterrows():
            prompt = user_content.format(row)
            prompt_list.append({
                'item_id': row['item_id'],
                GENERATED_SIG_COLNAME: row[GENERATED_SIG_COLNAME],
                JUDGE_PROMPT_COLNAME: prompt
            })

        return pd.DataFrame(prompt_list)


def create_train_test_datasets_multi(annotated_df, train_size=0.7, how='assistant_only_label',
                                     hdfs_out_dirname=None,
                                     hdfs_out_dir_base='/user/b_perso/mmandelbrod/gen_sigs/',
                                     override=False,
                                     random_state=None):
    if how not in prompts:
        raise ValueError(f"`how` should be one of {list(prompts.keys())}")

    is_multi = prompts[how].get('is_multi', False)
    annotated_df = annotated_df[~annotated_df[LABEL_COLNAME].isna()]

    # Split by item_id if multi-signal
    if is_multi:
        unique_items = annotated_df['item_id'].unique()
        train_ids, test_ids = train_test_split(unique_items, train_size=train_size, random_state=42)
        df_train = annotated_df[annotated_df['item_id'].isin(train_ids)]
        df_test = annotated_df[annotated_df['item_id'].isin(test_ids)]
    else:
        df_train, df_test = train_test_split(annotated_df, train_size=train_size, random_state=42)

    df_messages_train = create_athena_dataset_multi(df_train, how=how)
    df_messages_test = create_inference_dataset_multi(df_test, how=how)

    if hdfs_out_dirname is not None:
        hdfs_write_path = os.path.join(hdfs_out_dir_base, hdfs_out_dirname)
        write_to_hdfs(df_messages_train, 'train_athena.parquet', hdfs_write_path)
        write_to_hdfs(df_messages_test, 'test.parquet', hdfs_write_path)

    return df_messages_train, df_messages_test


def create_kfold_datasets(
        annotated_df,
        n_splits=3,
        how='assistant_only_label',
        hdfs_out_dir_base='/user/b_perso/mmandelbrod/gen_sigs/',
        base_hdfs_dirname='kcv_run',
        write_to_hdfs_flag=False
):
    """
    K-Fold CV dataset creator (single or multi signal).
    Writes each fold to:
      .../base_hdfs_dirname_fold{n}_train/train_athena.parquet
      .../base_hdfs_dirname_fold{n}_test/test.parquet
    Prints full HDFS file paths if written.
    """
    if how not in prompts:
        raise ValueError(f"`how` should be one of: {list(prompts.keys())}")

    is_multi = prompts[how].get('is_multi', False)
    annotated_df = annotated_df[~annotated_df[LABEL_COLNAME].isna()]
    unique_items = annotated_df['item_id'].unique()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_datasets = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_items), start=1):
        train_ids = unique_items[train_idx]
        test_ids = unique_items[test_idx]

        df_train = annotated_df[annotated_df['item_id'].isin(train_ids)]
        df_test = annotated_df[annotated_df['item_id'].isin(test_ids)]

        if is_multi:
            df_messages_train = create_athena_dataset_multi(df_train, how=how)
            df_messages_test = create_inference_dataset_multi(df_test, how=how)
        else:
            df_messages_train = create_athena_dataset_one_sig_per_item(df_train, how=how)
            df_messages_test = create_inference_dataset_one_sig_per_item(df_test, how=how)

        if write_to_hdfs_flag:
            fold_dir = os.path.join(hdfs_out_dir_base, f"{base_hdfs_dirname}_fold{fold_idx}_train")

            train_file = os.path.join(fold_dir, 'train_athena.parquet')
            test_file = os.path.join(fold_dir, 'test.parquet')

            write_to_hdfs(df_messages_train, out_filename='train_athena.parquet', out_dir=fold_dir)
            write_to_hdfs(df_messages_test, out_filename='test.parquet', out_dir=fold_dir)

            # print(f"[Fold {fold_idx}] Saved train → {train_file}")
            # print(f"[Fold {fold_idx}] Saved test  → {test_file}")

        fold_datasets.append((df_messages_train, df_messages_test))

    return fold_datasets


def init_chat(base_model, adapter):
    chat = EbayLLMChatWrapper(
        model_name=base_model,
        model_adapter=adapter,
        max_tokens=4096,
        temperature=0,
        top_p=0.98,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    return chat


def extract_label_one_sig_per_item(text):
    match = re.search(r"Label:\s*(\d)", text)
    if match:
        return match.group(1)
    return None


def classify_gensigs_one_sig_per_item(gensigs_df, base_model, adapter):
    chat = init_chat(base_model, adapter)
    responses = []
    for i, row in gensigs_df.iterrows():
        curr_prompt = row[JUDGE_PROMPT_COLNAME]
        print(curr_prompt)
        curr_response = chat([HumanMessage(content=curr_prompt)])
        # curr_score = str(curr_response).split('Label: ')[1].split(" ' ")[0]
        curr_score = extract_label_one_sig_per_item(str(curr_response))
        print(f"curr_score: {curr_score}")
        responses.append(int(float(curr_score)))
    gensigs_df['pred_score'] = responses
    return gensigs_df


def extract_label_multi(text):
    match = re.findall(r"Label[s]*:\s*([0-9,\s]+)", text)
    if match:
        return [int(x.strip()) for x in match[0].split(',') if x.strip().isdigit()]
    return []


def clean_grouped_signals(text):
    """
    Cleans the grouped signals by removing numbering prefixes like '1. ' but keeps signal content intact.
    """
    if pd.isna(text):
        return []

    lines = str(text).split('\n')
    cleaned = []

    for line in lines:
        line = line.strip()
        # Remove "number." at the beginning (e.g., '1. ', '12. ', etc.)
        cleaned_line = re.sub(r'^\d+\.\s*', '', line)
        if cleaned_line:
            cleaned.append(cleaned_line.strip())

    return cleaned


def ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return [x]

# Inference:
def classify_gensigs_multi(gensigs_df, base_model, adapter, how):
    chat = init_chat(base_model, adapter)
    is_multi = prompts[how].get('is_multi', False)

    responses = []

    for i, row in gensigs_df.iterrows():
        prompt = row[JUDGE_PROMPT_COLNAME]
        print(f"[{i}] Prompt:\n{prompt}")
        response = chat([HumanMessage(content=prompt)])
        print(f"[{i}] Response:\n{response}")

        label_values = extract_label_multi(str(response))
        responses.append(label_values if is_multi else (label_values[0] if label_values else -1))

    gensigs_df['pred_score'] = responses

    if is_multi:
        # Ensure grouped_signals exist
        if 'grouped_signals' not in gensigs_df.columns:
            raise ValueError("Expected 'grouped_signals' column in multi-signal input.")

        # Clean grouped_signals into list of strings
        gensigs_df['grouped_signals_clean'] = gensigs_df['grouped_signals'].apply(clean_grouped_signals)
        gensigs_df['pred_score'] = gensigs_df['pred_score'].apply(ensure_list)

        # Create can_explode flag
        gensigs_df['can_explode'] = gensigs_df.apply(
            lambda row: int(len(row['grouped_signals_clean']) == len(row['pred_score'])),
            axis=1
        )

    else:
        gensigs_df['can_explode'] = 0  # single signal mode

    # Return summary table (no explode yet)
    if is_multi:
        return gensigs_df[['item_id', 'grouped_signals', JUDGE_PROMPT_COLNAME, 'pred_score', 'can_explode']]
    else:
        return gensigs_df[['item_id', GENERATED_SIG_COLNAME, JUDGE_PROMPT_COLNAME, 'pred_score', 'can_explode']]


def explode_predictions(df):
    """
    Explodes rows where 'can_explode' == 1 into multiple rows per signal + score.
    """
    if 'can_explode' not in df.columns:
        raise ValueError("'can_explode' column not found. Make sure to run classify_gensigs_multi first.")

    # Filter explodable rows only
    df_explodable = df[df['can_explode'] == 1].copy()

    # Ensure needed columns exist
    if 'grouped_signals_clean' not in df_explodable.columns:
        df_explodable['grouped_signals_clean'] = df_explodable['grouped_signals'].apply(clean_grouped_signals)
    if not isinstance(df_explodable['pred_score'].iloc[0], list):
        df_explodable['pred_score'] = df_explodable['pred_score'].apply(ensure_list)

    # Explode grouped_signals_clean and pred_score together
    df_exploded = df_explodable.explode(['grouped_signals_clean', 'pred_score']).reset_index(drop=True)

    # Optional: Rename for clarity
    df_exploded = df_exploded.rename(columns={
        'grouped_signals_clean': 'signal',
        'pred_score': 'label'
    })

    return df_exploded[['item_id', 'signal', 'label', JUDGE_PROMPT_COLNAME]]


def join_predictions_to_annotated_df(df_agg_fashion_hg, df_messages_test_exp):
    """
    Joins predicted labels from df_messages_test_exp to df_agg_fashion_hg
    using item_id and signal text (case-insensitive match).

    Returns:
        Merged DataFrame with 'pred_score' column.
    """
    df_agg_fashion_hg['gen_sig_gpt_cleaned'] = df_agg_fashion_hg['gen_sig_gpt'].str.lower().str.strip()
    df_messages_test_exp['signal_cleaned'] = df_messages_test_exp['signal'].str.lower().str.strip()

    merged = df_agg_fashion_hg.merge(
        df_messages_test_exp,
        left_on=['item_id', 'gen_sig_gpt_cleaned'],
        right_on=['item_id', 'signal_cleaned'],
        how='inner')

    if 'label_y' in merged.columns:
        merged.rename(columns={'label_y': 'pred_score'}, inplace=True)
    elif 'label' in merged.columns:
        merged.rename(columns={'label': 'pred_score'}, inplace=True)

    return merged


#Mistral:
def init_model_mistral(base_model_name, adapter_name, torch_dtype=torch.bfloat16):
    """
    Initializes Mistral model and tokenizer with PEFT adapter.
    """
    config = PeftConfig.from_pretrained(adapter_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="balanced", torch_dtype=torch_dtype)
    model = PeftModel.from_pretrained(model, adapter_name)
    return tokenizer, model


def extract_label_multi_mistral(text):
    import re
    match = re.findall(r"Label[s]*:\s*([0-9,\s]+)", text)
    if match:
        last_match = match[-1]
        return [int(x.strip()) for x in last_match.split(',') if x.strip().isdigit()]
    return []


def classify_gensigs_multi_mistral(
        gensigs_df, tokenizer, model, how, max_new_tokens=128, temperature=0.001
):
    is_multi = prompts[how].get('is_multi', False)
    responses = []

    for i, row in gensigs_df.iterrows():
        prompt = row[JUDGE_PROMPT_COLNAME]

        print(f"\n[{i}] Prompt:\n{prompt}")

        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature
        )
        decoded = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"[{i}] Response:\n{decoded}")

        label_values = extract_label_multi_mistral(str(decoded))
        responses.append(label_values if is_multi else (label_values[0] if label_values else -1))

    gensigs_df['pred_score'] = responses

    if is_multi:
        if 'grouped_signals' not in gensigs_df.columns:
            raise ValueError("Expected 'grouped_signals' column in multi-signal input.")

        gensigs_df['grouped_signals_clean'] = gensigs_df['grouped_signals'].apply(clean_grouped_signals)
        gensigs_df['pred_score'] = gensigs_df['pred_score'].apply(ensure_list)

        gensigs_df['can_explode'] = gensigs_df.apply(
            lambda row: int(len(row['grouped_signals_clean']) == len(row['pred_score'])),
            axis=1
        )
    else:
        gensigs_df['can_explode'] = 0

    if is_multi:
        return gensigs_df[['item_id', 'grouped_signals', JUDGE_PROMPT_COLNAME, 'pred_score', 'can_explode']]
    else:
        return gensigs_df[['item_id', GENERATED_SIG_COLNAME, JUDGE_PROMPT_COLNAME, 'pred_score', 'can_explode']]


def fix_single_signal_multi_predictions(df):
    """
    If num_signals == 1 and pred_score is a list of length > 1,
    trim pred_score down to just the first prediction.
    """

    def fix(row):
        if 'grouped_signals' in row and isinstance(row['pred_score'], list):
            num_signals = row['grouped_signals'].count('\n') + 1 if '\n' in row['grouped_signals'] else 1
            if num_signals == 1 and len(row['pred_score']) > 1:
                return [row['pred_score'][0]]
        return row['pred_score']

    dfc = df.copy()
    dfc['pred_score'] = dfc.apply(fix, axis=1)
    return dfc


def init_model_llama(base_model_name, adapter_name, torch_dtype=torch.float16):
    """
    Initializes LLaMA model and tokenizer with PEFT adapter using safer device map.
    """
    from ellement.transformers import AutoTokenizer, AutoModelForCausalLM
    from ellement.peft import PeftModel, PeftConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Use "sequential" or "balanced_low_0" to spread layers across GPUs more conservatively
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="sequential",  # or "balanced_low_0" if you want better balance
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )

    # Attach adapter (same device map applies)
    model = PeftModel.from_pretrained(model, adapter_name)

    return tokenizer, model


def classify_gensigs_multi_llama(gensigs_df, tokenizer, model, how, max_new_tokens=128, temperature=0.01):
    is_multi = prompts[how].get('is_multi', False)
    responses = []

    for i, row in gensigs_df.iterrows():
        prompt = row[JUDGE_PROMPT_COLNAME]
        print(f"\n[{i}] Prompt:\n{prompt}")

        inputs = tokenizer(prompt, return_tensors='pt')
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature
        )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"[{i}] Response:\n{decoded}")

        label_values = extract_label_multi_mistral(str(decoded))  # reuse the same extractor if format is same
        responses.append(label_values if is_multi else (label_values[0] if label_values else -1))

    gensigs_df['pred_score'] = responses

    if is_multi:
        if 'grouped_signals' not in gensigs_df.columns:
            raise ValueError("Expected 'grouped_signals' column in multi-signal input.")

        gensigs_df['grouped_signals_clean'] = gensigs_df['grouped_signals'].apply(clean_grouped_signals)
        gensigs_df['pred_score'] = gensigs_df['pred_score'].apply(ensure_list)

        gensigs_df['can_explode'] = gensigs_df.apply(
            lambda row: int(len(row['grouped_signals_clean']) == len(row['pred_score'])),
            axis=1
        )
    else:
        gensigs_df['can_explode'] = 0

    return gensigs_df[['item_id', 'grouped_signals', JUDGE_PROMPT_COLNAME, 'pred_score', 'can_explode']]

