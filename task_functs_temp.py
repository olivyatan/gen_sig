import asyncio
import os
import sys
import pandas as pd
sys.path.append('src')
sys.path.append('dev/generation')


model_name = 'mistral'
business_vertical = 'pna'
data_path = '/data/shpx/data/mmandelbrod/GenSigs/'
infilepath =  f'Datasets/hag_pna_collct_09_24/ExtractedAmended/{business_vertical}_us_09_27_amended.csv' # 'Datasets/hag_pna_collct_09_24/ExtractedAmended/pna_us_09_27_amended.csv'
outfile_with_forbidden = f'{business_vertical}_with_forbidden_{model_name}'
outfile_no_forbidden = f'{business_vertical}_no_forbidden_{model_name}'

outpath_tmp_json = os.path.join(data_path, 'Datasets/hag_pna_collct_09_24/GenSigsData/JsonTemp')
outpath_final = os.path.join(data_path, 'Datasets/hag_pna_collct_09_24/GenSigsData')


def generate():
    print(f"sys.path: {sys.path}")
    print(f"current dir: {os.getcwd()}")
    print(f"listdir: {os.listdir()}")
    fullpath = os.path.abspath(__file__)
    print(f"current file path: : {fullpath}. __file__: {__file__}")
    dirname = os.path.dirname(fullpath)
    os.chdir( dirname )
    print("After chdir()")
    print(f"current dir: {os.getcwd()}")
    print(f"listdir: {os.listdir()}")
    print(f"listdir('src'): {os.listdir('src')}")
    print('hi. In generate()')
    sys.path.append(os.path.join(dirname, 'src'))
    sys.path.append(os.path.join(dirname, 'dev/generation'))
    print(f"sys.path: {sys.path}\n")
    with open(os.path.join(data_path, 'pykry_experiment.txt'), 'w') as f:
        f.write(f"Strating generate()\n")
        f.write(f"sys.path: {sys.path}\n")
        f.write(f"current dir: {os.getcwd()}\n")
        f.write(f"listdir: {os.listdir()}\n")

        f.write('hi. In generate()\n')

    from generation.sigs_gen import generate_signals_per_item
    print('imported generate_signals_per_item')
    print(f"data_path: {data_path}")
    infile_path = os.path.join(data_path, infilepath)
    print(f"infile_path: {infile_path}")
    df_amended_raw = pd.read_csv( infile_path, index_col=0)
    print(f"len(df_amended_raw): {len(df_amended_raw)}")

    # Generate signals without filtering forbidden words

    df_with_sigs_with_forbidden = asyncio.run(generate_signals_per_item(
        df_amended_raw,
        cols_for_prompt=['title', 'aspects', 'desc'],
        outfile=f'{outpath_tmp_json}/{outfile_with_forbidden}.jsonl',
        model_name=model_name
    ))
    df_with_sigs_with_forbidden.to_parquet(f'{outpath_final}/{outfile_with_forbidden}.parquet')
    print('done generating with forbidden')

    # Generate signals over filtered forbidden words
    df_with_sigs_no_forbidden =  asyncio.run(generate_signals_per_item(
        df_amended_raw,
        cols_for_prompt=['title_no_forbidden','aspects_no_forbidden','desc_no_forbidden'],
        outfile=f'{outpath_tmp_json}/{outfile_no_forbidden}.jsonl',
        model_name=model_name
    ))
    df_with_sigs_no_forbidden.to_parquet(f'{outpath_final}/{outfile_no_forbidden}.parquet')
    print('done with generate()')

# def train():
#     with open(os.path.join(outdir_path, 'pykry_experiment.txt'), 'w') as f:
#         f.write(f"train model")

def validate():
    print(f"sys.path: {sys.path}")
    print(f"current dir: {os.getcwd()}")
    print(f"listdir: {os.listdir()}")
    print(f"Hi. In validate()")

    with open(os.path.join(data_path, 'pykry_experiment.txt'), 'a+') as f:
        f.write(f" validate model")
