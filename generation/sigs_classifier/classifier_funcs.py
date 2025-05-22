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
  - 45In. X 3.6 Yds
  - 40 Pieces Per Pack
  - Four Sizes And Options
  - Compatible Part Number
  - USB Or Battery Powered
  - 2-Pc. 5-In. Gnomes
  - Genuine Samsung Da29-00003G
  - 18V Lithium 2Ah Battery
  - Fits 26 - 50 Lbs
  - 50 X 60 In
  - 2100 # 122444 5 Lumens

  Label 1 (Good Signal): The signal is clear, relevant, and provides useful information about the product, which is not predent in the product's title.
  Examples:
  - 97% Natural Ingredients
  - Pristine Condition
  - Smooth Rotating Design
  - Durable Stainless Steel
  - Multi-Sizing Options
  - USDA Organic Certified
  - Non-Stick Surface
  - Strong Wind Speed 

  Label 2 (Very Good Signal):** In addition to being a good signal (1) the signal incentivizes the user to purchase.
  Examples:
  - Natural Solid Wood Material
  - Great Condition
  - 5 Years Manufacturer Warranty
  - Solar Powered
  
Here is the listing data:
\n Item title: {{0}}, \n Item aspects: {{1}}, \n Item description: {{2}}. \n Generated signal: {{3}}
'''

ASSISTANT_ONLY_LABEL_CONTENT = f"Label: {{0}}"
ASSISTANT_CONTENT_LABEL_AND_REASON = assistant_content = f"Label: {{0}}, Reason for label: {{1}}"
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
            format_list = [str(row[field]).strip() for field in self.fields_names_to_extract]
        elif how == 'numerical_label':
            format_list = [str(row[field]).strip() if field != LABEL_COLNAME else int(float(row[field])) for field in self.fields_names_to_extract]
        elif how == 'numerical_binary_label':
            res = []
            for field in self.fields_names_to_extract:
                if field != LABEL_COLNAME:
                    res.append(str(row[field]).strip())
                else:
                    label_val = int(float(row[field]))
                    res.append(1 if label_val != 0 else 0)
            format_list = [str(row[field]).strip() if field != LABEL_COLNAME else int(float(row[field])) for field in self.fields_names_to_extract]
        else:
            raise ValueError(f"how should be one of ['numerical_label', None]")
        return self.content_format_str.format(*format_list)


prompts = {

'assistant_only_label': {
    'user_content': PromptFormattedStr(USER_CONTENT_BASIC_EXPLANATION,
                                        [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, GENERATED_SIG_COLNAME]),
    'assistant_content': PromptFormattedStr(ASSISTANT_ONLY_LABEL_CONTENT, [LABEL_COLNAME])},

'assistant_only_label_ortal_prompt': {
    'user_content': PromptFormattedStr(USER_CONTENT_EXPLANATION_ORTAL,
                                        [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, GENERATED_SIG_COLNAME]),
    'assistant_content': PromptFormattedStr(ASSISTANT_ONLY_LABEL_CONTENT, [LABEL_COLNAME])
},

'assistant_label_and_explanation': {
    'user_content': PromptFormattedStr(USER_CONTENT_BASIC_EXPLANATION,
                                        [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, GENERATED_SIG_COLNAME]),
    'assistant_content': PromptFormattedStr(ASSISTANT_ONLY_LABEL_CONTENT, [LABEL_COLNAME, REASON_COLNAME + COMMENT_COLNAME])},

'assistant_label_and_single_reason_ortal_prompt': {
    'user_content': PromptFormattedStr(USER_CONTENT_EXPLANATION_ORTAL,
                                        [TITLE_COLNAME, ASPECTS_COLNAME, DESCRIPTION_COLNAME, GENERATED_SIG_COLNAME]),
    'assistant_content': PromptFormattedStr(ASSISTANT_CONTENT_LABEL_AND_REASON, [LABEL_COLNAME, MANUAL_REASONS_COLNAME])}
}


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

def create_athena_dataset(annotated_df, how='assistant_only_label'):
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
    df_messages = pd.DataFrame({"messages": slst })
    return df_messages

def create_inference_dataset(df_test, how):
    if how not in prompts.keys():
        raise ValueError(f"how should be one of {prompts.keys()}")
    user_content = prompts[how]['user_content']
    prompt_list = []

    for index, row in df_test.iterrows():
        curr_prompt = user_content.format(row)
        prompt_list.append(curr_prompt)
    df_test[JUDGE_PROMPT_COLNAME] = prompt_list
    return df_test

def create_train_test_datasets(annotated_df, train_size=0.7, how='assistant_only_label',
                               hdfs_out_dirname=None,
                               hdfs_out_dir_base='/user/b_perso/mmandelbrod/gen_sigs/',
                               override=False
                               ):
    annotated_df = annotated_df[~annotated_df[LABEL_COLNAME].isna()]
    df_train, df_test = train_test_split(annotated_df, train_size=0.7)
    df_messages_train = create_athena_dataset(df_train, how=how)
    df_messages_test = create_inference_dataset(df_test, how=how)
    if hdfs_out_dirname is not None:
        hdfs_write_path = os.path.join(hdfs_out_dir_base, hdfs_out_dirname)
        write_to_hdfs(df_messages_train, out_filename='train_athena.parquet', out_dir=hdfs_write_path)
        write_to_hdfs(df_messages_test, out_filename='test.parquet', out_dir=hdfs_write_path)
    return df_messages_train, df_messages_test


def init_chat(base_model, adapter):

    chat = EbayLLMChatWrapper(
        model_name=base_model,
        model_adapter=adapter,
        max_tokens=4600,
        temperature=0.2,
        top_p=0.98,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    return chat


def extract_label(text):
    match = re.search(r"Label:\s*(\d)", text)
    if match:
        return match.group(1)
    return None
def classify_gensigs(gensigs_df, base_model, adapter):
    chat = init_chat(base_model, adapter)
    responses = []
    for i, row in gensigs_df.iterrows():
        curr_prompt = row[JUDGE_PROMPT_COLNAME]
        print(curr_prompt)
        curr_response = chat([HumanMessage(content=curr_prompt)])
        # curr_score = str(curr_response).split('Label: ')[1].split(" ' ")[0]
        curr_score = extract_label(str(curr_response))
        print(f"curr_score: {curr_score}")
        responses.append(int(float(curr_score)))
    gensigs_df['pred_score'] = responses
    return gensigs_df
