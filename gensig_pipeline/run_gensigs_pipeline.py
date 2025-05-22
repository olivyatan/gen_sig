import argparse
import jsonpickle

import sys
import os
import matplotlib.pyplot as plt

# print(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'workflows'))
# print( f"sys.path: {sys.path}")
from pipelines.xgb_predict_signal_pipeline import XGBPredictSigPipeline
from pipelines.fixed_ranking_pipeline import FixedRankingPipeline
from workflows.workflow_utils import WorkflowConstants
from workflows.pipelines.causalml_decision_tree_pipeline import CausalMLPipeline

from global_utils.config import GenericConfig, load_config_dict




def basic_sigs_generator(xgb_ser: str, run_id: str, is_local_run: bool = False, save_files: bool = False,
                force_save: bool = False):
    print("Starting xgb_trainer...")
    root_gc: GenericConfig = jsonpickle.loads(xgb_ser)
    root_gc.is_local_run = is_local_run
    root_gc.raw['run_id'] = run_id
    print("creating pipeline")
    gensig_pipeline = GenSigPipeline(root_gc)
    print("staring pipeline")
    pipeline_objects = gensig_pipeline.execute_pipeline()
    pipeline_objects.conf = xgb_ser
    if save_files:
        pipeline_objects.save(gensig_pipeline.output_dir_base, run_id, exclude_fields=['raw_data_df', 'prepared_data'],
                              force_save=force_save)
    return pipeline_objects




def main():
    tasks_dict = {'generate_basic': basic_sigs_generator  }

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='task type', required=True, choices=tasks_dict.keys())
    parser.add_argument('--config-path', type=str, help='config file', required=True)
    parser.add_argument('--project-name', type=str, help='project name', required=False, default=None)
    parser.add_argument('--run_local', action='store_true', help='If specified - run locally instead of krylov',
                        required=False, default=False)
    parser.add_argument('--run_id', type=str, help='unique run id for stored results', required=False,
                        default=WorkflowConstants.run_id_not_specified.value)
    # parser.add_argument('--stages_to_execute', nargs='*', help='which pipeline stages to execute', required=False,
    #                     default=None,
    #                     choices=['load_data', 'prepare_data', 'generate_features_and_labels', 'split_train_test',
    #                              'train_model', 'predict_on_test', 'generate_metrics_and_plots']
    #                     )
    parser.add_argument('--modify_conf', nargs='*',
                        help='If specified, replace corresponding values in conf file per each element in the provided list',
                        required=False,
                        default=None,
                        )
    parser.add_argument('--save_files', action='store_true', help='If speficied - save all pipeline objects to files',
                        required=False, default=False)
    parser.add_argument('--force_save', action='store_true', help='If speficied - save all pipeline objects to files, '
                                                                  'override without alert', required=False,
                        default=False)

    args = parser.parse_args()

    config_dict = load_config_dict(args.config_path)
    root_gc = GenericConfig(config_dict)
    conf_modifications = args.modify_conf
    if conf_modifications is not None:
        root_gc.modify_conf(conf_modifications)
    assert args.task in tasks_dict.keys(), f"{args.task} is not a supported task, select one of {tasks_dict.keys()}"
    if args.run_id is None:
        print("run_id is mandatory for saving results. If you don't want to save results, run with --run_id=none")

    gc_ser = jsonpickle.dumps(root_gc)
    task_args = [gc_ser, args.run_id, args.run_local, args.save_files, args.force_save]

    if args.run_local:
        task_object = tasks_dict[args.task]
        pipeline_objects = task_object(*task_args)
        plt.show()
        return pipeline_objects
    else:
        # gc_ser = jsonpickle.dumps(root_gc)
        from global_utils.krylov import Krylovizator
        kry = Krylovizator(root_gc)
        # task_args = [gc_ser]
        job_id = kry.submit_job(task_object=tasks_dict[args.task], task_args=task_args, project_name=args.project_name)
        print(f"Launched the job. Job id: {job_id}")


if __name__ == "__main__":
    pipeline_objects = main()



'''
Runs examples: 


'''

