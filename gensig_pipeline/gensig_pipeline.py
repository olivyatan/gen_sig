#
# # from data_wrangling import process_feature
# import multiprocessing
# import os
# import time
# from datetime import datetime
# import sys
# from dev.generation.sigs_gen import generate_signals_per_item
# import numpy as np
#
# # sys.path.append('/home/mmandelbrod/repositories/bx_signals')
# from gensig_pipeline_base import *
#
# # from sklearn.utils import compute_sample_weight
# # from bx_signals.models_code.xgb_predict_sig_model import XGBPredictSigModel
# #
# # from composite_trtmnt_extractor import CompositeTreatmentExtractor
# # # sys.path.append('/Users/mmandelbrod/workspace/signals/src')
# # # sys.path.append('/Users/mmandelbrod/workspace/signals/workflows')
# # # from xgb.xgb_funcs import generate_uplift_curve_and_auc
# # from dataset_utils import reduce_data_df, explode_df, get_features_columns, calc_df_uplift
# # from global_utils.config import GenericConfig
# # import pipelines.causalml_decision_tree_pipeline as causalml_decision_tree_pipeline
# # from src.metrics.xgb_metrics_funcs import generate_recall_precision_metrics_multiclass, generate_roc_curve_mutliclass, \
# #     generate_features_importance, generate_metrics_for_multi_classifier
# #
# # from pipelines.signals_pipeline_base import SignalsTrainingPipelineBase, SignalsTrainingPipelineBaseConfig
# # import pandas as pd
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
# # import matplotlib.pyplot as plt
# # import xgboost as xgb
# # import shap
# # from functools import partial
# # from parallel_pandas import ParallelPandas
# # from metrics.auuc_funcs import generate_uplift_curve_and_auc, generate_conversion_metrics_no_nsh
# #
# #
# # # from causalml.metrics.visualize import *
# # # import numpy as np
# # # from global_utils.krylov_config import KrylovConfig
# # from causalml.dataset import make_uplift_classification
# #
# # from causalml.metrics import plot_gain, plot_qini
# # from causalml.dataset import make_uplift_classification
# # from workflows.src.utils import save, load
# # import numpy as np
# #
# # from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
# # from joblib import dump, load
# # from sklearn.metrics import auc
#
# '''
# Trains a single XGBoost model and estimates its performance. The modes are:
# 2. Predict signal (multi class classifier): features and bbowac label as a feature. Label: the signal
# 3. Predict signal (multi class classifier - Yuri's suggestion): features Label: the signal + the bbowac label.
# '''
#
# # TODO: check treated with no signal shown. Is it only due to non eligibility?
#
# class XGBPredictSigPipelineConfig(GenSigsPipelineBaseConfig):
#     def __init__(self, root_gc: GenericConfig):
#         super(XGBPredictSigPipelineConfig, self).__init__(root_gc)
#         gc = root_gc.get_man_gc('xgb_pipeline')
#         self.training_approach = gc.get_man('training_approach')
#         self.training_label_mode = gc.get_man("training_label_mode")
#         self.use_control_set_for_training = gc.get_man("use_control_set_for_training")
#         self.label_cols = gc.get_man('label_cols')
#         self.conversion_colname = gc.get_man('conversion_colname')
#         self.treatment_indx_colname = gc.get_man('treatment_indx_colname')
#         self.control_or_treated_colname = gc.get_man('control_or_treated_colname')
#         self.reduce_dataset = gc.get_man('reduce_dataset')
#         self.n_shown_colname = gc.get_man("n_shown_colname")
#         self.n_qualified_colname = gc.get_man("n_qualified_colname")
#         self.composite_qualified_colname = gc.get_man("composite_qualified_colname")
#         self.composite_shown_colname = gc.get_man("composite_shown_colname")
#         self.conversion_colname = gc.get_man("conversion_colname")
#         self.control_name = gc.get_man("control_name")
#         self.remove_no_quals = gc.get_man("remove_no_quals")
#         self.treatment_name = gc.get_man("treatment_name")
#         self.num_top_quals = gc.get_man('num_top_quals')
#         self.shown_thresh = gc.get_man('shown_thresh')
#         self.use_conversion_as_feature = gc.get_man('use_conversion_as_feature')
#         self.use_only_conversions_for_training = gc.get_man('use_only_conversions_for_training')
#         self.keep_converted_and_percent_train = float(gc.get('keep_converted_and_percent_train', -1))
#         self.parallelize = gc.get_man('parallelize')
#         self.xgb_objective = gc.get_man('xgb_objective')
#         self.stratify_train_test_split = gc.get_man('stratify_train_test_split')
#         self.xgb_params = gc.get_man('xgb_params')
#
#
#
#         # self.metrics = gc.get_man('metrics')
#         self.show_based_on_xgb_model = gc.get_man('show_based_on_xgb_model')
#         self.comp_quals_to_apply = gc.get('comp_quals_to_apply')
#         self.scale_pos_weight = gc.get_man('scale_pos_weight')
#         self.scale_nsh_weight = gc.get_man('scale_nsh_weight')
#         self.scale_non_choiceable_weight = float(gc.get('scale_non_choiceable_weight', default=1))
#         self.dont_use_nsh = gc.get('dont_use_nsh', default=False)
#         self.chunk_size = int(gc.get('chunk_size', default='-1'))
#         self.sample_weight_method = gc.get('sample_weight_method')
#         self.train_only_over_choiceable = gc.get('train_only_over_choiceable', default=False)
#         metrics_gc = gc.get_man_gc('metrics')
#         self.normalize_auuc = metrics_gc.get_man('normalize_auuc')
#         self.plot_lift = metrics_gc.get('plot_lift', default=False)
#         self.produce_train_metrics = metrics_gc.get('produce_train_metrics', default=False)
#         self.produce_test_metrics = metrics_gc.get('produce_test_metrics', default=True)
#         self.produce_metrics_for_base_classifier = metrics_gc.get('produce_metrics_for_base_classifier', default=True)
#         self.produce_metrics_over_choiceable_only = metrics_gc.get('produce_metrics_over_choiceable_only', default=True)
#
#
#
#         # If variations are specified - overriede default parameters
#         super(XGBPredictSigPipelineConfig, self).override_variation(root_gc)
#
#
#
#
#
# class GenSigPipeline(GenSigsPipelineBase):
#
#     def __init__(self, root_gc: GenericConfig):
#         super(GenSigsPipelineBase, self).__init__(root_gc)
#         self.__dict__.update(self.conf.__dict__)
#
#
#     def get_my_confstructor(self):
#         return GenSigsPipelineBase
#
#     def get_my_constructor(self):
#         return GenSigPipeline
#
#     def load_data(self, pipeline_objects):
#
#         super(self.__class__, self).load_data(pipeline_objects)
#         base_path  = self.base_path
#
#
#         infile_path = os.path.join(base_path, infilepath)
#         print(f"infile_path: {infile_path}")
#         df_amended_raw = pd.read_csv(infile_path, index_col=0)
#         print(f"len(df_amended_raw): {len(df_amended_raw)}")
#
#         # Generate signals without filtering forbidden words
#
#         df_with_sigs_with_forbidden = asyncio.run(generate_signals_per_item(
#             df_amended_raw,
#             cols_for_prompt=['title', 'aspects', 'desc'],
#             outfile=f'{outpath_tmp_json}/{outfile_with_forbidden}.jsonl',
#             model_name=model_name
#         ))
#         df_with_sigs_with_forbidden.to_parquet(f'{outpath_final}/{outfile_with_forbidden}.parquet')
#         print('done generating with forbidden')
#
#         # Generate signals over filtered forbidden words
#         df_with_sigs_no_forbidden = asyncio.run(generate_signals_per_item(
#             df_amended_raw,
#             cols_for_prompt=['title_no_forbidden', 'aspects_no_forbidden', 'desc_no_forbidden'],
#             outfile=f'{outpath_tmp_json}/{outfile_no_forbidden}.jsonl',
#             model_name=model_name
#         ))
#         df_with_sigs_no_forbidden.to_parquet(f'{outpath_final}/{outfile_no_forbidden}.parquet')
#         print('done with generate()')
#
#         dont_use_nsh = self.dont_use_nsh
#         train_only_over_choiceable = self.train_only_over_choiceable
#         sig_not_shown_identifier = self.get_signals_conf_property('sig_not_shown_identifier')
#         confounder_colnames = pipeline_objects.get_field('pp_signals_quals_colnames')
#         print("before reading the parquet file")
#         # Both fillna and reset_index are due to poor saving. once this is fixed I can omit this.
#         print(f"Reading the parquet file {self.train_data_path}")
#         signals_composite_df = pd.read_parquet(self.train_data_path)
#         # if \
#         # self.train_data_path.endswith('.parquet') else pd.read_csv(self.train_data_path, index_col=0)
#         print("after reading the parquet file")
#         if dont_use_nsh:
#             print("Removing rows with no signal shown")
#             signals_composite_df = signals_composite_df
#                 [signals_composite_df[self.composite_shown_colname] != sig_not_shown_identifier]
#         if train_only_over_choiceable:
#             len_before = len(signals_composite_df)
#             print("Keeping only choiceable rows")
#             signals_composite_df = signals_composite_df[signals_composite_df[confounder_colnames].sum(axis=1) > 1]
#             print(f"removed {len_before - len(signals_composite_df)} "
#                   f"({10 0 *(len_befor e -len(signals_composite_df) ) /len_before:.2f}%)"
#                   f" rows with no choiceable signal shown")
#
#         non_categorical_colnames = [c for c in signals_composite_df.columns if
#                                     not isinstance(signals_composite_df[c].dtype, pd.CategoricalDtype)]
#         signals_composite_df[non_categorical_colnames] = signals_composite_df[non_categorical_colnames].fillna(0)
#         print("after assigning the cols")
#
#         # TODO: This should be improved. For the feature colnames we should keep a type in the schema
#         signals_composite_df = signals_composite_df.apply(pd.to_numeric, errors='ignore')
#         pipeline_objects.raw_data_df = signals_composite_df.reset_index(drop=True)
#         print(f"loaded raw_df which has {len(signals_composite_df)} rows")
#
#         pipeline_objects.exclude_save_fields.append('raw_data_df')
#
#
#
#
#     def prepare_data(self, pipeline_objects ):
#
#         causalml_decision_tree_pipeline.CausalMLPipeline.prepare_data(self, pipeline_objects)
#
#
#     def generate_features_and_labels(self, pipeline_objects ):
#         super(self.__class__, self).generate_features_and_labels(pipeline_objects)
#         signals_df = pipeline_objects.get_field('prepared_data')
#         configured_featurenames = self.get_signals_conf_property('features')
#         # TODO - make this more robust
#         configured_pp_signals = self.get_signals_conf_property('placements.pp.signals')
#         pp_featurename_format = self.get_signals_conf_property('tracking_col_format.qualified')
#         composite_qualified_colname = self.composite_qualified_colname
#         conversion_colname = self.conversion_colname
#         composite_shown_colname = self.composite_shown_colname
#         configured_pp_signals_quals = list(map(lambda x: pp_featurename_format.format(x), configured_pp_signals))
#         features_cols = get_features_columns(signals_df, self.featureset, configured_featurenames,
#                                              configured_pp_signals_quals)
#         # features_cols_for_metrics = get_features_columns(signals_df, self.featureset_for_metrics, configured_featurenames,
#         #                                      configured_pp_signals_quals)
#         # Include all features (metrics and features) in the features columns. When training and predicting - use only the subset.
#         cols_to_add = [f for f in configured_pp_signals_quals if f not in features_cols]
#
#
#         sig_not_shown_identifier = self.get_signals_conf_property('sig_not_shown_identifier')
#         orig_sql_index_colname = self.orig_sql_index_colname
#
#
#         conversion_1_colname = None
#         features_cols_for_inference = None
#         training_label_colname = 'training_label'
#         if self.training_label_mode == 'composite_sig':  # Label is (composite signal)
#             signals_df[training_label_colname] = signals_df[composite_shown_colname]
#             features_cols_for_training = features_cols + [conversion_colname]
#             conversion_1_colname = 'conversion_1_col'
#             signals_df[conversion_1_colname] = True
#             features_cols_for_inference = features_cols + [conversion_1_colname]
#         if self.training_label_mode == 'composite_sig_yuri': # Label is (composite signal, conversion)
#             signals_df[training_label_colname] = signals_df[composite_shown_colname] + ',' + signals_df
#                 [conversion_colname].astype(int).astype(str)
#             one_conversion_labels_str = signals_df[training_label_colname][signals_df[conversion_colname] == 1].unique()
#             nsh_conversion_labels_str = signals_df[training_label_colname]
#                 [signals_df[composite_shown_colname] == sig_not_shown_identifier].unique()
#             features_cols_for_training = features_cols
#
#         res_df = signals_df[[orig_sql_index_colname] +
#                             ['event_timestamp', 'item_id', 'user_id'] \
#                             + features_cols_for_training \
#                             + ([conversion_colname] if conversion_colname not in features_cols_for_training else []) \
#                             + [composite_qualified_colname, composite_shown_colname] \
#                             + [self.control_or_treated_colname] + [training_label_colname] \
#                             + ([conversion_1_colname] if conversion_1_colname is not None else  []) \
#                             + cols_to_add # Make sure the qualification indicators are in the dataframe.
#                             ]
#
#
#         # #['label'] is the original df label - converted y/n. self.get_lable_cols() depends on the pipeline's configuration.
#         # res_df = signals_df[features_cols  + [composite_qualified_colname, 'treatment'] + list(set( [conversion_colname] + self.get_label_cols(True)))]
#         # TODO: below should be updated. It's for the case where the label is the composite signal, so I'm adding 'control' as that signal.
#         # SHould be kept as a separate conlumn
#         # res_df[self.get_label_cols()][signals_df['treatment'] == 'control'] = 'control'
#
#         label_encoder = LabelEncoder()
#         numeric_labels = label_encoder.fit_transform(res_df[training_label_colname])
#         training_label_colname_numeric = training_label_colname + "_numeric"
#         res_df[training_label_colname_numeric] = numeric_labels
#         if self.training_label_mode == 'composite_sig_yuri':
#             one_conversion_labels_num = label_encoder.transform(one_conversion_labels_str)
#             nsh_conversion_labels_num = label_encoder.transform(nsh_conversion_labels_str)
#
#
#
#         pipeline_objects.features_cols_for_training = features_cols_for_training
#         pipeline_objects.features_cols_with_quals = features_cols_for_training + cols_to_add
#         pipeline_objects.label_encoder = label_encoder
#         pipeline_objects.training_label_colname = training_label_colname
#         pipeline_objects.training_label_colname_numeric = training_label_colname_numeric
#         pipeline_objects.features_labels_df = res_df
#         pipeline_objects.pp_signals_quals_colnames = configured_pp_signals_quals
#         pipeline_objects.actual_treatment_colnanme = composite_shown_colname
#         pipeline_objects.features_cols_for_inference = features_cols_for_inference
#         pipeline_objects.conversion_1_colname = conversion_1_colname
#
#         if self.training_label_mode == 'composite_sig_yuri':
#             pipeline_objects.one_conversion_labels_num = one_conversion_labels_num
#             pipeline_objects.nsh_conversion_labels_num = nsh_conversion_labels_num
#             pipeline_objects.one_conversion_labels_str = one_conversion_labels_str
#
#
#
#
#     # TODO! Important - when splitting a learning-to-rank dataframe, I want to keep the groups. If it's pointwise it's not
#     # Necessary, but it's best to implement later on.
#     def split_train_test(self, pipeline_objects):
#         super(self.__class__, self).split_train_test(pipeline_objects)
#         df_train_test_labels = pipeline_objects.get_field('features_labels_df')
#         use_only_conversions_for_training = self.use_only_conversions_for_training
#         keep_converted_and_percent_train = self.keep_converted_and_percent_train
#         if use_only_conversions_for_training and keep_converted_and_percent_train != -1:
#             raise Exception('use_only_conversions_for_training and keep_converted_and_percent_train'
#                             ' are mutually exclusive')
#         split_by = self.conf.split_by
#         # split_by_item_id = self.conf.split_by_item_id
#         # TODO - add options to the configuration file (by date, test+validation). Also - this may split in the middle of the day, so metrics are biased.
#         # Maybe keep granularity of days.
#         if split_by == 'date':
#             if keep_converted_and_percent_train != -1:
#                 raise Exception('keep_converted_and_percent_train is not supported when splitting by' \
#                                 ' date since it will cause leakages' )
#             print("splitting by date. prop_train: ", self.prop_train)
#             df_train_test_labels = df_train_test_labels.sort_values('event_timestamp', ascending=True)
#             print(f"len(df_train_test_labels): {len(df_train_test_labels)}")
#             print(f"len(df_train_test_labels) * self.prop_train: {len(df_train_test_labels) * self.prop_train}")
#             n_rows_for_train = int(len(df_train_test_labels) * self.prop_train)
#             df_train = df_train_test_labels.iloc[:n_rows_for_train]
#             df_test = df_train_test_labels.iloc[n_rows_for_train:]
#             print(f"Splitting by date. Train size: {len(df_train)}, Test size: {len(df_test)}")
#             # print the dates of each
#             from_date = datetime.fromtimestamp(df_train['event_timestamp'].min( ) /1000000000).date()
#             to_date = datetime.fromtimestamp(df_train['event_timestamp'].max() / 1000000000).date()
#             print(f"Train dates: {from_date} - {to_date}")
#             i f(len(df_test) > 0):
#                 from_date_test = datetime.fromtimestamp(df_test['event_timestamp'].min() / 1000000000).date()
#                 to_date_test = datetime.fromtimestamp(df_test['event_timestamp'].max() / 1000000000).date()
#                 print(f"Test dates: {from_date_test} - {to_date_test}")
#         elif split_by == 'item_id':
#             print("splitting by item_id")
#             # Assuming df is your DataFrame and 'item_id' is the column you want to split by
#             unique_item_ids = df_train_test_labels['item_id'].unique()
#
#             # Split the unique item ids into train and test sets
#             print(f"Splitting using random state {self.random_state}")
#             train_item_ids, test_item_ids = train_test_split(unique_item_ids, test_size= 1 -self.prop_train, \
#                                                              random_state=self.random_state)
#
#             # Create train and test dataframes based on the item ids
#             df_train = df_train_test_labels[df_train_test_labels['item_id'].isin(train_item_ids)]
#             df_test = df_train_test_labels[df_train_test_labels['item_id'].isin(test_item_ids)]
#         elif split_by == 'user_id':
#             print("splitting by user_id")
#             unique_user_ids = df_train_test_labels['user_id'].unique()
#
#             # Split the unique user ids into train and test sets
#             print(f"Splitting using random state {self.random_state}")
#             train_user_ids, test_user_ids = train_test_split(unique_user_ids, test_size= 1 -self.prop_train, \
#                                                              random_state=self.random_state)
#
#             # Create train and test dataframes based on the user ids
#             df_train = df_train_test_labels[df_train_test_labels['user_id'].isin(train_user_ids)]
#             df_test = df_train_test_labels[df_train_test_labels['user_id'].isin(test_user_ids)]
#         else:
#             raise Exception(f"Unknown split_by value: {split_by}")
#
#         if use_only_conversions_for_training:
#             print(f"Using only conversions for training")
#             len_before = len(df_train)
#             df_train_only_conversions = df_train[df_train[self.conversion_colname] == 1]
#             print(f"Removed {len_before - len(df_train_only_conversions)} rows with no conversions")
#             print(f"Train size after removing non-conversions: {len(df_train_only_conversions)}")
#             print()
#             pipeline_objects.df_train = df_train_only_conversions
#         elif keep_converted_and_percent_train != -1 :
#             print(f"Keeping only {keep_converted_and_percent_train} of the non converted rows for training")
#             converted_rows = df_train[df_train[self.conversion_colname] == 1]
#             non_converted_rows = df_train[df_train[self.conversion_colname] == 0]
#             non_concerted_to_keep = int(len(converted_rows) * keep_converted_and_percent_train)
#             df_train_reduced = pd.concat([converted_rows, non_converted_rows.sample(n=non_concerted_to_keep,
#                                                                                     random_state=self.random_state)])
#             print(f"Initial train size: {len(df_train)}. Reduced to {len(df_train_reduced)}. "
#                   f"Converted: {len(converted_rows)}, Non-converted to keep: {non_concerted_to_keep}")
#             pipeline_objects.df_train = df_train_reduced
#
#         else:
#             pipeline_objects.df_train = df_train
#
#         # if not use_external_test_df:
#         pipeline_objects.df_test = df_test
#         # else:
#         #     print(f"Switching test df to be the external one: {self.external_test_df}")
#         #     external_test_df = pd.read_parquet(self.external_test_df)
#         #     #Make sure all columns are there
#         #     if not set(df_test.columns).issubset(set(external_test_df.columns)):
#         #         raise Exception("The external test df doesn't include all the expected columns!")
#         #     external_test_df[composite_shown_colname] = external_test_df[composite_shown_colname].astype(str)
#         #     pipeline_objects.df_test = external_test_df
#         #     pipeline_objects.used_external_test_df = self.external_test_df
#
#
#
#     def my_compute_sample_weight(self, df_train,
#                                  signals_quals_colnames,
#                                  sig_shown_colname,
#                                  conversion_colname,
#                                  scale_pos_weight,
#                                  scale_nsh_weight,
#                                  scale_non_choiceable_weight,
#                                  sig_not_shown_identifier,
#                                  sample_weight_method = 'per_qualification_set'):
#         print("In my_compute_sample_weight!!")
#         print(f"sample_weight_method: {sample_weight_method}")
#         sample_weights = None
#         res_df = pd.DataFrame({'sample_weight': np.ones(len(df_train))}, index=df_train.index)
#         # Per qualification set, we want all qualified signals to have the same weight,
#         # So the model learns a non-biased frequency between them. This is important mainly for the 'nsh' signals, which
#         # is largen than the others (equals their sum). If there's no 'nsh' in the data and randomization is proper, the
#         # shown signals are balanced in every qualification set anyhow.
#         if sample_weight_method == 'per_qualification_set':
#             for confouns_val, confouns_val_df in df_train.groupby(signals_quals_colnames):
#                 curr_sample_weights = compute_sample_weight(
#                     class_weight='balanced',
#                     y=confouns_val_df[sig_shown_colname]
#                 )
#                 # Additionally, scale up the positive samples (conversions), since we want to learn these better.
#                 mask = confouns_val_df[conversion_colname] == 1
#                 curr_sample_weights[mask] *= scale_pos_weight
#
#                 # Scale down the nsh samples, so that the models will predict then only with high confidence
#                 mask_nsh = confouns_val_df[sig_shown_colname] == sig_not_shown_identifier
#                 curr_sample_weights[mask_nsh] *= scale_nsh_weight
#                 res_df.loc[confouns_val_df.index, 'sample_weight'] = curr_sample_weights
#
#             sample_weights = res_df['sample_weight'].values
#
#         elif sample_weight_method == 'by_labels':
#             # prioritize the converted rows over the non-converted ones
#             sample_weights = compute_sample_weight(
#                 class_weight='balanced',
#                 y=df_train[conversion_colname]
#             )
#             mask = df_train[conversion_colname] == 1
#             sample_weights[mask] *= scale_pos_weight
#         elif sample_weight_method == 'none':
#             sample_weights = np.ones(len(df_train))
#         elif sample_weight_method == 'only_pos':
#             print(f"Scaling only pos weights")
#             mask = df_train[conversion_colname] == 1
#             sample_weights = mask * scale_pos_weight
#             print(f"sample_weights: {np.unique(sample_weights, return_counts=True)}. mean: {sample_weights.mean()}")
#         elif sample_weight_method == 'ivp':
#             pass
#             # divide the set (x,y,z) by p(x|z)
#             for confounder_val, confounder_df in df_train.groupby(signals_quals_colnames):
#                 for x_z_val, x_z_df in confounder_df.groupby( sig_shown_colname ):
#                     p_x_given_z = len(x_z_df) / len(confounder_df)
#                     res_df.loc[x_z_df.index, 'sample_weight'] = 1/ p_x_given_z
#             # TODO: for the time being, I'm keeping these out for the IVP case. Consider adding them later.
#             # # Additionally, scale up the positive samples (conversions), since we want to learn these better.
#             # mask = confouns_val_df[conversion_colname] == 1
#             # curr_sample_weights[mask] *= scale_pos_weight
#             #
#             # # Scale down the nsh samples, so that the models will predict then only with high confidence
#             # mask_nsh = confouns_val_df[sig_shown_colname] == sig_not_shown_identifier
#             # curr_sample_weights[mask_nsh] *= scale_nsh_weight
#             # res_df.loc[confouns_val_df.index, 'sample_weight'] = curr_sample_weights
#             print(f"Scaling pos weights")
#             mask = df_train[conversion_colname] == 1
#             res_df.loc[mask, 'sample_weight'] = res_df.loc[mask, 'sample_weight'] * scale_pos_weight
#             sample_weights = res_df['sample_weight'].values
#
#         print(f"Scaling down non-choiceable rows ({scale_non_choiceable_weight})")
#         non_choiceable_inds = df_train[signals_quals_colnames].sum(axis=1) <= 1
#         sample_weights[non_choiceable_inds] *= scale_non_choiceable_weight
#
#         return sample_weights
#
#     def train_model(self, pipeline_objects):
#         # raise Exception("This method should not be called. Use the train_model_with_cv method instead")
#         super(self.__class__, self).train_model(pipeline_objects)
#         start_time = time.time()
#
#         df_train = pipeline_objects.get_field('df_train')
#         label_colname = pipeline_objects.get_field('training_label_colname_numeric')
#         featurs_cols_for_training = pipeline_objects.get_field('features_cols_for_training')
#         conversion_colname = self.conversion_colname
#         if self.training_label_mode == 'composite_sig_yuri':
#             nsh_conversion_labels_num = pipeline_objects.get_field('nsh_conversion_labels_num')
#
#         scale_nsh_weight = self.scale_nsh_weight
#         scale_pos_weight = self.scale_pos_weight
#         scale_non_choiceable_weight = self.scale_non_choiceable_weight
#         xgb_params = self.xgb_params
#         sig_shown_colname = self.composite_shown_colname
#         signals_quals_colnames = pipeline_objects.get_field('pp_signals_quals_colnames')
#         sample_weight_method = self.sample_weight_method
#         sig_not_shown_identifier = self.get_signals_conf_property('sig_not_shown_identifier')
#
#         x_train, y_train = df_train[featurs_cols_for_training].astype(float), \
#             df_train[label_colname].astype(int)
#
#         # counts = pd.value_counts(y_train)
#         # scale_pos_weight = counts[0] / (1.5 * counts[1])
#         # print(f"scale_pos_weight: {scale_pos_weight}")
#
#         if self.training_label_mode == 'composite_sig_yuri':
#             sample_weights = compute_sample_weight(
#                 class_weight='balanced',
#                 y=y_train  # provide your own target name
#             )
#             # If we choose balanced weights, the model will predict more or less the same number of 1s and 0s between the
#             # treatments and the 'nsh' treatment. (since the 'nsh' treatment is the most common one, it will be predicted often.)
#             # In fact, it may be a good idea to train only over the treated only, since I know that rarely does 'nsh' has higher uplift.
#             # In the below I'll just reduce its weight.
#             mask = np.isin(y_train, nsh_conversion_labels_num)
#             sample_weights[mask] *= scale_nsh_weight
#             # print(f"sample_weights: {sample_weights}")
#             # xgb_params['num_class'] = len(df_train[label_colname].unique())
#             # print(f"num_class: {xgb_params['num_class']}")
#             # print(f"objective: {xgb_params['objective']}")
#         elif self.training_label_mode == 'composite_sig':
#             sample_weights = self.my_compute_sample_weight(df_train,
#                                                            signals_quals_colnames,
#                                                            sig_shown_colname,
#                                                            conversion_colname,
#                                                            scale_pos_weight,
#                                                            scale_nsh_weight,
#                                                            scale_non_choiceable_weight,
#                                                            sig_not_shown_identifier,
#                                                            sample_weight_method=sample_weight_method
#                                                            )
#
#
#
#         else:
#             print(f"Unknown training label mode {self.training_label_mode}")
#         xgb_params['num_class'] = len(df_train[label_colname].unique())
#         print(f"num_class: {xgb_params['num_class']}")
#         print(f"objective: {xgb_params['objective']}")
#         # print(f"sample_weights: {np.unique(sample_weights, return_counts=True)}")
#         chunk_size = self.chunk_size if self.chunk_size != -1 else len(x_train) + 1
#         print(f"Chunk size: {chunk_size}")
#         model = self.train_model_incrementally(x_train, y_train, xgb_params, sample_weights,
#                                                featurs_cols_for_training, chunk_size)
#
#         pipeline_objects.trained_model = model
#         pipeline_objects.trained_model_to_production = XGBPredictSigModel(model,
#                                                                           pipeline_objects.get_field('label_encoder'),
#                                                                           signals_quals_colnames,
#                                                                           self.get_signals_conf_property(
#                                                                               'tracking_col_format.qualified')
#                                                                           )
#         end_time = time.time()
#         print(f"Training the model took: {(end_time - start_time) // 60} minutes")
#
#     # Todo: should move to utils file
#     def qualified_colname_to_sigid(self, colname, sigids):
#         return [s for s in sigids if s in colname]
#
#     # Does two things:
#     # 1. Removes the (sig_id, 0) predictions (We want to show the signal which is most likely to produce conversion (i.e. 1)
#     # 2. Finds the signal with the highest probability of  1 conversion which is withing the instance's qualification set.
#     def amend_prediction_res_to_comply_yuri(self, predicted_signals_probs, label_encoder, qualification_df,
#                                             quals_colname_format, sig_not_shown_sigid, one_conversion_labels):
#         # Step 1
#         predicted_signals_probs_one = predicted_signals_probs[:, one_conversion_labels]
#         # Step 2
#         one_conversion_labels_str = label_encoder.inverse_transform(one_conversion_labels)
#         sigids = [x.split(',')[0] for x in one_conversion_labels_str]
#         pred_comp_prob = pd.DataFrame(predicted_signals_probs_one, columns=sigids)
#         # qualification_df = qualification_df.rename(columns =
#         #             {  colname : self.qualified_colname_to_sigid(colname, configured_pp_signals) for colname in\
#         #                qualification_df.columns})
#         qualification_df[quals_colname_format.format(sig_not_shown_sigid)] = 1  # Not showing always qualifies.
#         for colname in pred_comp_prob.columns:
#             pred_comp_prob.loc[(qualification_df[quals_colname_format.format(colname)] == 0).values, colname] = -1000
#         # Now take the maximal probability (non-qualified signals should have value -1000, and nhs should have value 0
#         predicted_signals_max_probs = pred_comp_prob.max(axis=1)  # np.argmax(predicted_signals_probs_one, axis=1)
#         predicted_signals_ids = pred_comp_prob.idxmax(axis=1)
#         return predicted_signals_max_probs, predicted_signals_ids
#
#     #  Finds the signal with the highest probability of  1 conversion which is within the instance's qualification set.
#     def amend_prediction_res_to_comply_label_as_feature(self, predicted_signals_probs, label_encoder, qualification_df,
#                                                         quals_colname_format,
#                                                         sig_not_shown_sigid,
#                                                         classes_names
#                                                         ):
#         pred_comp_prob = pd.DataFrame(predicted_signals_probs, columns=label_encoder.inverse_transform(classes_names))
#         qualification_df[quals_colname_format.format(sig_not_shown_sigid)] = 1  # Not showing always qualifies.
#         # Reduce the probability of a non-qualified signal to be shown to -1000
#         non_comp_val = -1000
#         for colname in pred_comp_prob.columns:
#             pred_comp_prob.loc[
#                 (qualification_df[quals_colname_format.format(colname)] == 0).values, colname] = non_comp_val
#         # Print how many changed vallue to -1000
#         print(f"Changed {(pred_comp_prob == non_comp_val).sum().sum()} values to {non_comp_val}")
#         # Now take the maximal probability (non-qualified signals should have value -1000, and nhs should have value 0
#         predicted_signals_max_probs = pred_comp_prob.max(axis=1)  # np.argmax(predicted_signals_probs_one, axis=1)
#         predicted_signals_ids = pred_comp_prob.idxmax(axis=1)
#
#         return predicted_signals_max_probs, predicted_signals_ids
#
#     def amend_prediction_res_to_comply(self, predicted_signals_probs, label_encoder, qualification_df,
#                                        quals_colname_format,
#                                        sig_not_shown_sigid,
#                                        classes_names,
#                                        one_conversion_labels=None):
#         if one_conversion_labels is None:
#             return self.amend_prediction_res_to_comply_label_as_feature(predicted_signals_probs, label_encoder,
#                                                                         qualification_df,
#                                                                         quals_colname_format,
#                                                                         sig_not_shown_sigid,
#                                                                         classes_names
#                                                                         )
#
#         else:
#             return self.amend_prediction_res_to_comply_yuri(predicted_signals_probs, label_encoder, qualification_df,
#                                                             quals_colname_format,
#                                                             sig_not_shown_sigid,
#                                                             one_conversion_labels)
#
#     def predict_on_df(self, df_to_predict, pipeline_objects):
#
#         trained_model = pipeline_objects.get_field('trained_model')
#         features_cols_for_inference = pipeline_objects.get_field('features_cols_for_inference')
#         features_cols_for_base_classifier_metrics = pipeline_objects.get_field('features_cols_for_training')
#         label_colname = pipeline_objects.get_field('training_label_colname_numeric')
#         label_encoder = pipeline_objects.get_field('label_encoder')
#         if self.training_label_mode == 'composite_sig_yuri':
#             one_conversion_labels_num = pipeline_objects.get_field('one_conversion_labels_num')
#         else:
#             one_conversion_labels_num = None
#         sig_not_shown_identifier = self.get_signals_conf_property('sig_not_shown_identifier')
#         pp_signals_colnames = pipeline_objects.get_field('pp_signals_quals_colnames')
#         pp_featurename_format = self.get_signals_conf_property('tracking_col_format.qualified')
#
#         x_test, y_test = df_to_predict[features_cols_for_inference].astype(float).values, \
#             df_to_predict[label_colname].astype(int).values
#         y_hat, y_prob = self.predict_in_chunks(trained_model, x_test)
#
#         predicted_signals_max_probs, predicted_signals_ids = self.amend_prediction_res_to_comply(y_prob,
#                                                                                                  label_encoder,
#                                                                                                  df_to_predict[
#                                                                                                      pp_signals_colnames],
#                                                                                                  pp_featurename_format,
#                                                                                                  sig_not_shown_identifier,
#                                                                                                  trained_model.classes_,
#                                                                                                  one_conversion_labels_num)
#
#         df_pred = df_to_predict.copy()  # TODO: do I need this?
#
#         df_pred['pred_comp_shown_numeric_before_compliance'] = y_hat
#         df_pred['pred_comp_shown_before_compliance'] = label_encoder.inverse_transform(y_hat)
#         df_pred['predicted_signals_max_probs_comply'] = predicted_signals_max_probs.values
#         df_pred['predicted_signals_ids_comply'] = predicted_signals_ids.values
#         classes_names_numeric = trained_model.classes_
#         df_pred.loc[:, classes_names_numeric] = y_prob
#
#         # In order to produce test metrics for the base classifier, I need to infer using the original
#         # train features
#
#         x_test_train_features, y_test_train_features = df_to_predict[features_cols_for_base_classifier_metrics].astype(
#             float).values, \
#             df_to_predict[label_colname].astype(int).values
#         y_hat_train_features, y_prob_train_features = self.predict_in_chunks(trained_model, x_test_train_features)
#
#         # y_hat_train_features = trained_model.predict(x_test_train_features)
#         # y_prob_train_features = trained_model.predict_proba(x_test_train_features)
#         df_pred_train_features = df_to_predict.copy()  # TODO: do I need this?
#         colnames = list(map(lambda s: f"p_{s}", label_encoder.inverse_transform(trained_model.classes_)))
#         pred_classes_prob = pd.DataFrame(y_prob_train_features, columns=colnames)
#         df_pred_train_features['pred_classes_numeric'] = y_hat_train_features
#         df_pred_train_features['pred_classes_str'] = label_encoder.inverse_transform(y_hat_train_features)
#         df_pred_train_features.loc[:,
#         pred_classes_prob.columns] = pred_classes_prob.values  # !! cautious - the indexes of the two dfs are different -> setting not the values will result in a nan sub-df
#         # df_pred_train_features['predicted_signals_ids_comply'] = predicted_signals_ids.values
#         # classes_names_numeric = trained_model.classes_
#         # df_pred.loc[:, classes_names_numeric] = y_prob
#         #
#
#         # TODO: note that this is from internal function, so this code is invoked twice - once on train and once on test df
#         #
#         pipeline_objects.pred_comp_prob_colname = 'predicted_signals_max_probs_comply'
#         pipeline_objects.predicted_treatment_colname = 'predicted_signals_ids_comply'
#         pipeline_objects.pred_classes_numeric_colname = 'pred_classes_numeric'
#         pipeline_objects.pred_classes_str_colname = 'pred_classes_str'
#         pipeline_objects.pred_classes_prob_colnames = colnames
#         return df_pred, df_pred_train_features
#
#     def predict_on_test(self, pipeline_objects):
#         super(self.__class__, self).predict_on_test(pipeline_objects)
#         composite_shown_colname = self.composite_shown_colname
#         external_test_df = self.external_test_df
#         use_external_test_df = (external_test_df) is not None and (external_test_df != 'None')
#         if not use_external_test_df:
#             df_to_predict = pipeline_objects.get_field('df_test')
#         else:
#             print(f"Switching test df to be the external one: {self.external_test_df}")
#             # TODO: from some reason the soft link is not read. I'll skip this test, implemente it later on (and read just the columns)
#             # curr_df_test = pipeline_objects.get_field('df_test')
#             external_test_df = pd.read_parquet(self.external_test_df)
#             # #Make sure all columns are there
#             # if not set(curr_df_test.columns).issubset(set(external_test_df.columns)):
#             #     raise Exception("The external test df doesn't include all the expected columns!")
#             external_test_df[composite_shown_colname] = external_test_df[composite_shown_colname].astype(str)
#             df_to_predict = external_test_df
#             pipeline_objects.df_test = external_test_df
#             pipeline_objects.used_external_test_df = self.external_test_df
#
#         full_pred_df_test, pred_df_base_features = self.predict_on_df(df_to_predict, pipeline_objects)
#         pipeline_objects.full_pred_df_test = full_pred_df_test
#         pipeline_objects.pred_df_base_features = pred_df_base_features
#
#         if self.produce_train_metrics:
#             df_to_predict = pipeline_objects.get_field('df_train')
#             full_pred_df_train, pred_df_base_features_train = self.predict_on_df(df_to_predict, pipeline_objects)
#             pipeline_objects.full_pred_df_train = full_pred_df_train
#             pipeline_objects.pred_df_base_features_train = pred_df_base_features_train
#
#     def generate_metrics_and_plots(self, pipeline_objects):
#         super(self.__class__, self).generate_metrics_and_plots(pipeline_objects)
#         result_df = pipeline_objects.get_field('full_pred_df_test')
#         pred_df_base_features = pipeline_objects.get_field('pred_df_base_features')
#         predicted_treatment_colname = pipeline_objects.get_field('predicted_treatment_colname')
#         actual_treatment_colnanme = pipeline_objects.get_field('actual_treatment_colnanme')
#         comp_prob_colname = pipeline_objects.get_field('pred_comp_prob_colname')
#         control_or_treated_colname = self.control_or_treated_colname
#         features_colnames = pipeline_objects.get_field('features_cols_with_quals')
#         trained_model = pipeline_objects.get_field('trained_model')
#         no_treatment_id = self.get_signals_conf_property('sig_not_shown_identifier')
#         plot_lift = self.plot_lift
#         produce_train_metrics = self.produce_train_metrics
#         produce_test_metrics = self.produce_test_metrics
#         produce_metrics_for_base_classifier = self.produce_metrics_for_base_classifier
#         normalize_auuc = self.normalize_auuc
#         sig_not_shown_identifier = self.get_signals_conf_property('sig_not_shown_identifier')
#         if produce_train_metrics:
#             result_df_train = pipeline_objects.get_field('full_pred_df_train')
#             pred_df_base_features_train = pipeline_objects.get_field('pred_df_base_features_train')
#         else:
#             result_df_train = None
#         actual_treatment_colnanme_numeric = pipeline_objects.get_field('training_label_colname_numeric')
#         actual_treatment_colnanme_str = pipeline_objects.get_field('training_label_colname')
#         conversion_colname = self.conversion_colname
#         pred_classes_numeric_colname = pipeline_objects.get_field('pred_classes_numeric_colname')
#         pred_classes_str_colname = pipeline_objects.get_field('pred_classes_str_colname')
#         pred_classes_prob_colnames = pipeline_objects.get_field('pred_classes_prob_colnames')
#         label_encoder = pipeline_objects.get_field('label_encoder')
#         pp_signals_quals_colnames = pipeline_objects.get_field('pp_signals_quals_colnames')
#         dont_use_nsh = self.dont_use_nsh
#         confounder_colnames = pipeline_objects.get_field('pp_signals_quals_colnames')
#
#         class_names = label_encoder.inverse_transform(trained_model.classes_)
#         print(f"produce_test_metrics: {produce_test_metrics}")
#         if produce_test_metrics:
#             print()
#             print('-----------------------------------------------------------')
#             print("Generating metrics and plots for the test dataframe")
#             print('-----------------------------------------------------------')
#             if dont_use_nsh:
#                 if not self.produce_metrics_over_choiceable_only:
#                     auuc_metrics_dict_no_nsh = generate_conversion_metrics_no_nsh(result_df,
#                                                                                   predicted_treatment_colname,
#                                                                                   actual_treatment_colnanme,
#                                                                                   conversion_colname,
#                                                                                   confounder_colnames,
#                                                                                   # pp_signals_quals_colnames
#                                                                                   comp_prob_colname,
#                                                                                   control_or_treated_colname,
#                                                                                   features_colnames,
#                                                                                   no_treatment_id,
#                                                                                   plot_normalized=False,
#                                                                                   plot_only_gain=True,
#                                                                                   control_value='control',
#                                                                                   pp_featurename_quals_format='sig_qualified_{}',
#                                                                                   sig_not_shown_identifier='nsh',
#                                                                                   should_calc_auuc_metrics=True)
#                     pipeline_objects.auuc_metrics_dict = auuc_metrics_dict_no_nsh
#
#                 print()
#                 print("Generating only over the choicable qualification signals")
#                 print()
#                 result_df_choiceable = result_df[result_df[pp_signals_quals_colnames].sum(axis=1) > 1]
#                 print(f"Percent of choicable rows: {len(result_df_choiceable) / len(result_df)}")
#                 print()
#                 auuc_metrics_dict_no_nsh_choiceable = generate_conversion_metrics_no_nsh(result_df_choiceable,
#                                                                                          predicted_treatment_colname,
#                                                                                          actual_treatment_colnanme,
#                                                                                          conversion_colname,
#                                                                                          confounder_colnames,
#                                                                                          # pp_signals_quals_colnames
#                                                                                          comp_prob_colname,
#                                                                                          control_or_treated_colname,
#                                                                                          features_colnames,
#                                                                                          no_treatment_id,
#                                                                                          plot_normalized=False,
#                                                                                          plot_only_gain=True,
#                                                                                          control_value='control',
#                                                                                          pp_featurename_quals_format='sig_qualified_{}',
#                                                                                          sig_not_shown_identifier='nsh',
#                                                                                          should_calc_auuc_metrics=True)
#
#                 pipeline_objects.auuc_metrics_dict_choiceable = auuc_metrics_dict_no_nsh_choiceable
#
#
#             else:
#                 if not self.produce_metrics_over_choiceable_only:
#                     auuc_metrics_dict = generate_uplift_curve_and_auc(result_df,
#                                                                       predicted_treatment_colname,
#                                                                       actual_treatment_colnanme,
#                                                                       conversion_colname,
#                                                                       comp_prob_colname,
#                                                                       control_or_treated_colname,
#                                                                       features_colnames,
#                                                                       no_treatment_id,
#                                                                       plot_normalized=normalize_auuc,
#                                                                       plot_only_gain=(not plot_lift),
#                                                                       sig_not_shown_identifier=sig_not_shown_identifier,
#                                                                       should_calc_auuc_metrics=self.calc_auuc_metrics
#                                                                       )
#                     pipeline_objects.auuc_metrics_dict = auuc_metrics_dict
#                 print()
#                 print("Generating only over the non-singleton qualification signals")
#                 print()
#                 result_df_no_singleton = result_df[result_df[pp_signals_quals_colnames].sum(axis=1) > 1]
#                 auuc_metrics_dict_choiceable = generate_uplift_curve_and_auc(result_df_no_singleton,
#                                                                              predicted_treatment_colname,
#                                                                              actual_treatment_colnanme,
#                                                                              conversion_colname,
#                                                                              comp_prob_colname,
#                                                                              control_or_treated_colname,
#                                                                              features_colnames,
#                                                                              no_treatment_id,
#                                                                              plot_normalized=normalize_auuc,
#                                                                              plot_only_gain=(not plot_lift),
#                                                                              sig_not_shown_identifier=sig_not_shown_identifier,
#                                                                              should_calc_auuc_metrics=self.calc_auuc_metrics
#                                                                              )
#
#                 pipeline_objects.auuc_metrics_dict_choiceable = auuc_metrics_dict_choiceable
#
#         if produce_train_metrics:
#             print()
#             print('-----------------------------------------------------------')
#             print("Generating metrics and plots for the train dataframe")
#             print('-----------------------------------------------------------')
#
#             if dont_use_nsh:
#                 auuc_metrics_dict_no_nsh = generate_conversion_metrics_no_nsh(result_df_train,
#                                                                               predicted_treatment_colname,
#                                                                               actual_treatment_colnanme,
#                                                                               conversion_colname,
#                                                                               confounder_colnames,
#                                                                               # pp_signals_quals_colnames
#                                                                               comp_prob_colname,
#                                                                               control_or_treated_colname,
#                                                                               features_colnames,
#                                                                               no_treatment_id,
#                                                                               plot_normalized=False,
#                                                                               plot_only_gain=True,
#                                                                               control_value='control',
#                                                                               pp_featurename_quals_format='sig_qualified_{}',
#                                                                               sig_not_shown_identifier='nsh',
#                                                                               should_calc_auuc_metrics=True)
#
#                 print()
#                 print("Generating only over the choicable qualification signals")
#                 print()
#                 result_df_train_choiceable = result_df_train[result_df_train[pp_signals_quals_colnames].sum(axis=1) > 1]
#                 print(f"Percent of choicable rows: {len(result_df_train_choiceable) / len(result_df)}")
#                 print()
#                 auuc_metrics_dict_no_nsh_choiceable = generate_conversion_metrics_no_nsh(result_df_train_choiceable,
#                                                                                          predicted_treatment_colname,
#                                                                                          actual_treatment_colnanme,
#                                                                                          conversion_colname,
#                                                                                          confounder_colnames,
#                                                                                          # pp_signals_quals_colnames
#                                                                                          comp_prob_colname,
#                                                                                          control_or_treated_colname,
#                                                                                          features_colnames,
#                                                                                          no_treatment_id,
#                                                                                          plot_normalized=False,
#                                                                                          plot_only_gain=True,
#                                                                                          control_value='control',
#                                                                                          pp_featurename_quals_format='sig_qualified_{}',
#                                                                                          sig_not_shown_identifier='nsh',
#                                                                                          should_calc_auuc_metrics=True)
#                 pipeline_objects.auuc_metrics_dict_train = auuc_metrics_dict_no_nsh
#                 pipeline_objects.auuc_metrics_dict_choiceable_train = auuc_metrics_dict_no_nsh_choiceable
#
#             else:
#
#                 auuc_metrics_dict_train = generate_uplift_curve_and_auc(result_df_train,
#                                                                         predicted_treatment_colname,
#                                                                         actual_treatment_colnanme,
#                                                                         conversion_colname,
#                                                                         comp_prob_colname,
#                                                                         control_or_treated_colname,
#                                                                         features_colnames,
#                                                                         no_treatment_id,
#                                                                         plot_normalized=normalize_auuc,
#                                                                         plot_only_gain=(not plot_lift),
#                                                                         sig_not_shown_identifier=sig_not_shown_identifier,
#                                                                         should_calc_auuc_metrics=self.calc_auuc_metrics
#                                                                         )
#
#                 pipeline_objects.auuc_metrics_dict_train = auuc_metrics_dict_train
#
#         if produce_metrics_for_base_classifier:
#             print()
#             print('-----------------------------------------------------------')
#             print("Generating metrics and plots for the base multi-class classifier")
#             print('-----------------------------------------------------------')
#             print()
#             if produce_test_metrics:
#                 print("------------------ Test set -------------------- ")
#                 self.generate_metrics_for_base_classifier(trained_model, pred_df_base_features,
#                                                           class_names,
#                                                           actual_treatment_colnanme_numeric,
#                                                           actual_treatment_colnanme_str,
#                                                           pred_classes_numeric_colname,
#                                                           pred_classes_str_colname,
#                                                           pred_classes_prob_colnames,
#                                                           conversion_colname)
#
#             if produce_train_metrics:
#                 print()
#                 print("------------------ Train set -------------------- ")
#                 print()
#                 self.generate_metrics_for_base_classifier(trained_model, pred_df_base_features_train,
#                                                           class_names,
#                                                           actual_treatment_colnanme_numeric,
#                                                           actual_treatment_colnanme_str,
#                                                           pred_classes_numeric_colname,
#                                                           pred_classes_str_colname,
#                                                           pred_classes_prob_colnames,
#                                                           conversion_colname)
#
#         print("---------- metrics summary ---------------")
#         print("Printing for choiceable only")
#         if produce_test_metrics:
#             print()
#             # print("----Test set: -----")
#             print()
#             # print(pipeline_objects.auuc_metrics_dict['summary_table'] )
#             print()
#             print("--------- Test set, only choiceable subset ------- ")
#             print()
#             print(pipeline_objects.auuc_metrics_dict_choiceable['summary_table'])
#             print()
#         if produce_train_metrics:
#             print("Train set:")
#             print(pipeline_objects.auuc_metrics_dict_choiceable_train['summary_table'])
#
#     # ## TODO: move to xgb mdtric fungs. Shouldn't be a class method.
#     # def generate_metrics_for_base_classifier_1(self, result_df,
#     #                                          actual_treatment_colnanme_numeric,
#     #                                          actual_treatment_colnanme_str,
#     #                                          pred_treatment_colnanme_numeric,
#     #                                          pred_treatment_colnanme_str,
#     #                                          pred_probs_colnames
#     #                                            ):
#     #
#     #     y_test_numeric = result_df[actual_treatment_colnanme_numeric].values
#     #     y_test_str = result_df[actual_treatment_colnanme_str].values
#     #     y_pred_numeric = result_df[pred_treatment_colnanme_numeric].values
#     #     y_pred_str = result_df[pred_treatment_colnanme_str]
#     #     y_pred_scores = result_df[pred_probs_colnames].values
#     #
#     #     print(f"Predicted v.s. actual distribution: ")
#     #     # actual_df_dist = pd.DataFrame({'actual':y_test_str}).value_counts(normalize=True).to_frame()
#     #     # pred_df_dist = pd.DataFrame({'predicted':y_pred_str}).value_counts(normalize=True).to_frame()
#     #     # print(pd.concat([actual_df_dist, pred_df_dist], axis=1, join='inner'))
#     #     actual_df_dist = pd.DataFrame({'actual': y_test_str}).value_counts(normalize=True).reset_index()
#     #     actual_df_dist.columns = ['value', 'actual']
#     #
#     #     pred_df_dist = pd.DataFrame({'predicted': y_pred_str}).value_counts(normalize=True).reset_index()
#     #     pred_df_dist.columns = ['value', 'predicted']
#     #
#     #     counts_df = pd.merge(actual_df_dist, pred_df_dist, on='value', how='inner')
#     #     print(counts_df)
#     #
#     #     generate_recall_precision_metrics_multiclass(y_test_numeric, y_pred_numeric, y_test_str, y_pred_str,
#     #                                                  y_pred_scores)
#
#     # Basic metrics for the base classifier, predicting the shown signal given that there was a conversion.
#     def generate_metrics_for_base_classifier(self, trained_model, result_df,
#                                              class_names,
#                                              actual_treatment_colnanme_numeric,
#                                              actual_treatment_colnanme_str,
#                                              pred_treatment_colnanme_numeric,
#                                              pred_treatment_colnanme_str,
#                                              pred_probs_colnames,
#                                              conversion_colname):
#
#         print()
#         print("------------ Over entire df -------------------")
#         print()
#         generate_metrics_for_multi_classifier(result_df,
#                                               class_names,
#                                               actual_treatment_colnanme_numeric,
#                                               actual_treatment_colnanme_str,
#                                               pred_treatment_colnanme_numeric,
#                                               pred_treatment_colnanme_str,
#                                               pred_probs_colnames
#                                               )
#
#         print()
#         print("------------ Over converted units -------------------")
#         print()
#         result_df_only_ones = result_df[result_df[conversion_colname] == 1]
#         generate_metrics_for_multi_classifier(result_df_only_ones,
#                                               class_names,
#                                               actual_treatment_colnanme_numeric,
#                                               actual_treatment_colnanme_str,
#                                               pred_treatment_colnanme_numeric,
#                                               pred_treatment_colnanme_str,
#                                               pred_probs_colnames
#                                               )
#
#         generate_features_importance(trained_model)
#
#     def get_label_cols(self, as_list=False):
#         if as_list:
#             if type(self.label_cols) == type([]):
#                 return self.label_cols
#             else:
#                 return [self.label_cols]
#         else:
#             if type(self.label_cols) == type([]):
#                 if (len(self.label_cols) > 1):
#                     print(
#                         f"Warning! get_label_cols() was invoked with as_list = {as_list}, but len(self.label_cols) = {len(self.label_cols)}")
#                 return self.label_cols[0]
#             else:
#                 return self.label_cols
#
#     def generate_recall_precision_metrics(self, result_df, actual_label_colname, predicted_label_colname,
#                                           predicted_prob_colbame):
#         y_test = result_df[actual_label_colname]
#         y_hat = result_df[predicted_label_colname]
#         y_prob = result_df[predicted_prob_colbame]
#
#         fpr, tpr, _ = roc_curve(y_test, y_prob)
#         roc_auc = auc(fpr, tpr)
#
#         # Plot ROC curve
#         plt.figure()
#         plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#         plt.plot([0, 1], [0, 1], 'k--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic')
#         plt.legend(loc="lower right")
#         plt.show(block=False)
#
#         # Compute recall, precision, and F1 score
#         recall = recall_score(y_test, y_hat)
#         precision = precision_score(y_test, y_hat)
#         f1 = f1_score(y_test, y_hat)
#
#         print(f"Recall: {recall}")
#         print(f"Precision: {precision}")
#         print(f"F1 Score: {f1}")
#
#     def generate_uplift_auc_score(self, result_df,
#                                   predicted_treatment_colname,
#                                   actual_treatment_colnanme,
#                                   conversion_colname,
#                                   comp_prob_colname,
#                                   control_or_treated_colname):
#         auuc_metrics, auc_score_random, auc_score_model, lift_df, gain_df = \
#             generate_uplift_curve_and_auc(
#                 result_df,
#                 predicted_treatment_colname,
#                 actual_treatment_colnanme,
#                 conversion_colname,
#                 comp_prob_colname,
#                 control_or_treated_colname
#             )
#
#         return auuc_metrics, auc_score_random, auc_score_model, lift_df, gain_df
#
#     '''
#     The input emulates the production input: a features vector, some of which are qualification indicators.
#     The goal is to predict the conversion probability for each individual qualified signal (separately, not as a composite signal)
#     Just to emphasize - here exploding is done based on qualified signals, not shown signals (as done for training)
#     '''
#
#     def generate_single_signals_matrix_from_composite(self, composite_matrix, shown_colnames, qual_colnames,
#                                                       add_no_treatment_row=False):
#         if len(shown_colnames) != len(qual_colnames):
#             raise Exception(f"len(shown_cols)={len(shown_colnames)} != len(qual_cols)={len(qual_colnames)}")
#
#         composite_matrix_for_train = composite_matrix.copy()
#         composite_matrix_for_train.loc[:, shown_colnames] = composite_matrix_for_train[qual_colnames].values
#         assert not (composite_matrix_for_train[shown_colnames] == composite_matrix[shown_colnames]).all().all()
#         # TODO: maybe add an empty row in the exploded df for the case: "show no signal"? If so - how to rank this with other signals?
#         exploded_df = explode_df(composite_matrix_for_train,
#                                  shown_colnames,
#                                  parallelize=self.parallelize,
#                                  num_threads=int(2 * multiprocessing.cpu_count() / 3),
#                                  add_no_treatment_row=add_no_treatment_row
#                                  )
#         return exploded_df
#
#     def pack_model_to_production(self, pipeline_objects):
#
#         trained_model = pipeline_objects.get_field('trained_model')
#         label_encoder = pipeline_objects.get_field('label_encoder')
#         pipeline_objects.model_to_production = {'model': trained_model, 'label_encoder': label_encoder}
#
#     def predict_in_chunks(self, trained_model, x_test, chunk_size=10000000):
#         y_hat = np.array([]).astype(int)
#         y_prob = np.empty((0, len(trained_model.classes_)))
#         num_chunks = x_test.shape[0] // chunk_size + 1
#         if num_chunks > 1:
#             print()
#             print(f"Predicting incrementally over {num_chunks} chunks")
#         for i in range(num_chunks):
#             print()
#             print(f"Chunk {i}")
#             print()
#             start_idx = i * chunk_size
#             end_idx = (i + 1) * chunk_size
#             x_test_chunk = x_test[start_idx:end_idx]
#             curr_y_hat = trained_model.predict(x_test_chunk)
#             curr_y_prob = trained_model.predict_proba(x_test_chunk)
#             # For the two classes case.
#             if len(curr_y_hat.shape) != 1:
#                 curr_y_hat = np.argmax(curr_y_prob, axis=1)
#             y_hat = np.concatenate([y_hat, curr_y_hat], axis=0)
#             y_prob = np.concatenate([y_prob, curr_y_prob], axis=0)
#         print(f"Finished predicting. Total num samples: {len(y_hat)}")
#         return y_hat, y_prob
#
#     def train_model_incrementally(self, x_train, y_train, xgb_params, sample_weights, featurs_cols_for_training,
#                                   chunk_size=10000000):
#
#         trained_model = None
#         num_chunks = len(x_train) // chunk_size + 1
#         if num_chunks > 1:
#             print(f"Training the model incrementally over {num_chunks} chunks")
#         for i in range(num_chunks):
#             print()
#             print(f"Chunk {i}")
#             print()
#             start_idx = i * chunk_size
#             end_idx = (i + 1) * chunk_size
#             x_train_chunk = x_train[start_idx:end_idx]
#             y_train_chunk = y_train[start_idx:end_idx]
#             sample_weights_chunk = sample_weights[start_idx:end_idx]
#
#             model = xgb.XGBClassifier(**xgb_params)
#             print("fitting model...")
#             model.fit(x_train_chunk, y_train_chunk,
#                       verbose=True,
#                       eval_set=[(x_train_chunk, y_train_chunk)],
#                       sample_weight=sample_weights_chunk,
#                       xgb_model=trained_model)
#             trained_model = model
#         trained_model._Booster.feature_names = featurs_cols_for_training
#         print()
#         print("finished")
#         print(f"num accumulated boosted rounds: {model.get_booster().num_boosted_rounds()}")
#         print()
#         return trained_model
#
#
# #
# # class XGBPredictSigModel():
# #     def __init__(self, xgb_model, label_encoder):
# #         self.xgb_model = xgb_model
# #         self.label_encoder = label_encoder
# #         self.sigs_ids = label_encoder.inverse_transform(self.xgb_model.classes_)
# #         self.model_filename = 'xgb_model.json'
# #         self.label_encoder_filename = 'label_encoder.joblib'
# #
# #     '''
# #     features_arr - assumed to be a 1d np.array with configured features and qualification features.
# #     '''
# #     #TODO: verify alignmemt of expected (model._Booster.feature_names ) and passed features.
# #     def predict(self, features_arr, qualification_features_indices):
# #         qualification_indicators = features_arr[:, qualification_features_indices]
# #         # Append the '1' feature
# #         features_arr = np.concatenate((features_arr, np.ones((features_arr.shape[0], 1))), axis=1)
# #         # Predict
# #         y_prob = self.xgb_model.predict_proba(features_arr)
# #         # Penalize non-compliant signals
# #         pred_comp_prob = pd.DataFrame(y_prob * qualification_indicators, columns= self.sigs_ids)
# #         # Select the compliant signal with the highest probability
# #         predicted_signal_id = pred_comp_prob.idxmax(axis=1)[0]
# #         return predicted_signal_id
# #
# #     def save(self, dirpath):
# #         os.makedirs(dirpath, exist_ok=True)
# #         self.xgb_model.save_model(os.path.join(dirpath, self.model_filename))
# #         dump(self.label_encoder, os.path.join(dirpath, self.label_encoder_filename) )
# #
# #
# #     def load(self, dirpath):
# #         # Create a new XGBoost classifier
# #         loaded_model = xgb.XGBClassifier()
# #         # Load the model from the file
# #         loaded_model.load_model(os.path.join(dirpath, self.model_filename))
# #         self.xgb_model = loaded_model
# #         self.label_encoder = load(os.path.join(dirpath, self.label_encoder_filename))
#
