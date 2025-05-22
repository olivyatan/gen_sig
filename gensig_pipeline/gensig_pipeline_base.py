import os
from abc import ABC, abstractmethod

# from causalml.inference.tree import UpliftRandomForestClassifier
from joblib import dump

import pandas

from global_utils.config import GenericConfig, load_config_dict
from global_utils.krylov_config import KrylovConfig
from pipeline_objects import PipelineObjects
from workflow_utils import WorkflowConstants
from collections import OrderedDict
from itertools import dropwhile
import traceback

import os.path as ospath
# from workflows.metrics_inputs import PipelineObjects


class GenSigsPipelineBaseConfig:
    def __init__(self, root_gc: GenericConfig):
        gc = root_gc.get_man_gc('gensig_base')
        self.run_id = root_gc.get('run_id')
        base_path_pykrylov = gc.get_man('base_path_pykrylov')
        base_path_bash = gc.get_man('base_path_bash')
        self.base_path = base_path_bash if root_gc.is_local_run else base_path_pykrylov
        self.output_dir_base = os.path.join( self.base_path, gc.get_man('output_dir_base') )
        self.train_data_path = os.path.join( self.base_path, gc.get_man('train_data_path') )
        self.prop_train = gc.get_man('prop_train')
        # TODO: This is not get_man_gc due to backward compatibility. Think of a better solution for this
        self.orig_sql_index_colname = gc.get('orig_sql_index_colname')
        self.signals_constants_path = gc.get_man('signals_constants_path')
        self.signals_conf_key = gc.get_man('signals_conf_key')
        self.featureset = gc.get_man('featureset')
        # self.featureset_for_metrics = gc.get('featureset_for_metrics', self.featureset)
        self.random_state = gc.get_man('random_state')
        self.max_num_sigs = gc.get_man('max_num_sigs')
        self.remove_leakages = gc.get_man('remove_leakages')
        self.split_by = gc.get_man('split_by')
        self.trainset_mode = gc.get('trainset_mode', 'all')
        self.subject_sig = gc.get('subject_sig', None)
        self.take_inputs_from = gc.get('take_inputs_from', None)
        self.calc_auuc_metrics = gc.get('calc_auuc_metrics', False)
        self.external_test_df = gc.get('external_test_df', None)



    def override_variation(self, root_gc: GenericConfig):
        gc = root_gc.get_gc('variations')
        if gc is not None:
            variation_list = gc.get('current_variation')
            if variation_list is not None:
                for var in variation_list:
                    curr_var = gc.get_man(var)
                    for param_name in curr_var.keys():
                        if '_path' in param_name:
                            self.__dict__[param_name] = os.path.join(self.base_path, curr_var[param_name])
                        else:
                            self.__dict__[param_name] = curr_var[param_name]



class GenSigsPipelineBase(ABC):

    def __init__(self, root_gc: GenericConfig):
        self.conf = self.get_my_confstructor()(root_gc)
        self.run_id = self.conf.run_id
        self.kry_conf = KrylovConfig(root_gc)
        self.output_dir_base = self.conf.output_dir_base
        self.train_data_path = self.conf.train_data_path
        self.signals_constants_path = self.conf.signals_constants_path
        self.signals_conf_key = self.conf.signals_conf_key
        self.prop_train = self.conf.prop_train
        self.featureset = self.conf.featureset
        self.random_state = self.conf.random_state
        self.orig_sql_index_colname = self.conf.orig_sql_index_colname
        self.signals_constants_conf = load_config_dict(self.signals_constants_path)['placements_sigs'][self.signals_conf_key]
        self.stages_od = OrderedDict()
        self.populate_stages_od(self.stages_od)

        # self.execute_pipeline()

    def update_conf(self, other_pipeline):
        self.__dict__.update(other_pipeline.__dict__)

    #TODO: implement a wrapper class to signals_constants_conf
    def get_signals_conf_property(self, key):
        keys = key.split('.')
        curr = self.signals_constants_conf
        for key in keys:
            curr = curr[key]
        return curr
    def get_my_confstructor(self):
        return GenSigsPipelineBaseConfig

    def write_object_to_file(self, obj, leafdir, outfile_name):
        outdir_path = os.path.join(self.output_dir_base, self.run_id if self.run_id is not None else '', leafdir)
        if not os.path.exists(outdir_path):
            os.makedirs(outdir_path)
        outfile_path = os.path.join(outdir_path, outfile_name)
        if os.path.exists(outfile_path):
            print(f"The file {outfile_path} exists. Should I overwrite it? (y/n)")
            answer = input()
            if answer == 'n':
                print("Exiting without overwriting.")
                exit(-1)
        if type(obj) == type(pandas.DataFrame()):
            obj.to_csv(outfile_path)
        else:
            dump(obj, outfile_path)

    '''
    @:param start_stage: The stage to start the pipeline from. If not specified, the pipeline will start from the first stage
    @:param stages_to_execute: A list of stages to execute. If not specified, the pipeline will execute all stages
    '''
    def execute_pipeline(self, start_stage = None, stages_to_execute = [], pipeline_obj = None):
        if len(stages_to_execute) > 0:
            start_stage = None
            print("!!! Attention !!")
            print(f"Since stages_to_execute is specified, ignoring start_stage")
        if start_stage is not None:
            if 'start_stage' in self.__dict__:
                print("There's already a start_stage specified! Overriding it with the new one.")
            self.start_stage = start_stage
        if pipeline_obj is None:
            pipeline_objects = PipelineObjects(self.output_dir_base, self.run_id)
            pipeline_objects.pipeline_constructor = self.get_my_constructor()
        else:
            pipeline_objects = pipeline_obj
        pipeline_objects.exclude_save_fields = []
        # Load the cached data to the pipeline
        if 'take_inputs_from' in self.__dict__:
            # pipeline_objects.base_path = self.base_path
            # if self is not None:
            #     pipeline_objects.cached_pipeline_path = self.take_inputs_from
            if self.take_inputs_from != 'None':
                pipeline_objects.load_from_files_dict(self.take_inputs_from, self.base_path)

        print(f"before starting pipeline. split_by: {self.split_by}")
        od_to_execute = self.stages_od
        if len(stages_to_execute) > 0:
            od1 = OrderedDict()
            for stage in od_to_execute.keys():
                if stage not in stages_to_execute:
                    print(f"removing {stage}")
                else:
                    print(f"Keeping stage {stage}")
                    od1[stage] = od_to_execute[stage]
            od_to_execute = od1


        # Define the key to start from
        start_key = list(od_to_execute)[0] if 'start_stage' not in self.__dict__ else self.start_stage

        # Create an iterator that skips keys until the start key
        iter_from_start_key = dropwhile(lambda key: key != start_key, od_to_execute.keys())

        # Iterate over the keys from the start key
        for key in iter_from_start_key:
            try:
                od_to_execute[key](pipeline_objects)
            except Exception as e:
                print()
                print("------------------------------------")
                print("!!!!Exception in pipeline execution!!!!!")
                print(f"Exception in {key}: {e}")
                print(f"An error occurred: {e}\n{traceback.format_exc()}")
                print()
                print("Exiting...")
                print()
                return pipeline_objects

        return pipeline_objects

    def populate_stages_od(self, od):
        self.od_to_execute["load_data"] = self.load_data
        self.od_to_execute["prepare_data"] = self.prepare_data
        self.od_to_execute["generate_signals"] = self.generate_signals
        self.od_to_execute["append_signals"] = self.append_signals
        self.od_to_execute["generate_metrics_and_plots"] = self.generate_metrics_and_plots

    @abstractmethod
    def get_my_constructor(self):
        pass

    @abstractmethod
    def load_data(self, pipeline_objects):
        print('')
        print('---------  load_data  ------------')
        print()

    @abstractmethod
    def prepare_data(self, pipeline_objects):
        print('')
        print('----------  prepare_data  -----------')
        print('')
        print()

    @abstractmethod
    def generate_signals(self, pipeline_objects):
        print('')
        print('--------- generate_signals ------------')
        print()

    @abstractmethod
    def append_signals(self, pipeline_objects):
        print('')
        print('----------  append_signals  -----------')
        print()


    @abstractmethod
    def generate_metrics_and_plots(self, pipeline_objects):
        print('')
        print('---------  generate_metrics_and_plots  ------------')
        print()


    def should_save_objects(self):
        return self.run_id != WorkflowConstants.run_id_not_specified.value