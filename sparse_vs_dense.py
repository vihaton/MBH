from collections import OrderedDict
import os
from datetime import datetime
from copy import deepcopy
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from scripts.utils import *
from scripts.data import get_train_and_test_loaders
from scripts.models import *
from scripts.pruning import *
from scripts.activation_patterns import *
from scripts.stats import *
from scripts.activation_regions import *
from scripts.lp_decision_trees import *


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run experiments sparse vs dense networks.')
    parser.add_argument('config_files', metavar='config_file', type=str, nargs='+',
                        help='paths to json file(s) that configure experiments to be run')
    parser.add_argument('--verbose', dest='verbose', action='store_const',
                        const=True, default=False,
                        help='Should we tell more what is going on?')

    args = parser.parse_args()

    # TODO if file name is directory, parse all configs from that dir
    configs = {}
    for cf in args.config_files:
        with open(cf, 'r') as fo:
            config = json.loads(fo.read())
            configs[cf.split('/')[-1]] = config

    return configs, args.verbose


def get_sdict_files(models_dir, verbose=False):
    assert os.path.isdir(
        models_dir), f'models dir {models_dir} didnt exist'

    return get_files_in_dir(models_dir, verbose=verbose)


def is_root_dir(models_dir):
    dirs = get_folders(models_dir)
    if dirs is None:
        return False

    sdict_files = get_sdict_files(models_dir, verbose=False)
    if len(sdict_files) > 0:
        for fname in sdict_files:
            if '.pt' == fname[-3:]:
                return False

    # we have dirs and we dont have .pt files
    return True


def run_experiments_models(config, verbose=False):
    # Models

    models_dir = config['models'].get('models_dir', '../mnist/models/')
    models_dir = add_trailing_slash(models_dir)

    sdict_files = get_sdict_files(models_dir, verbose)
    dirs = get_folders(models_dir)

    assert len(sdict_files) > 0 or len(
        dirs) > 0, 'there were no files or dirs! Did you configure the model directory correctly?'

    root_dir = is_root_dir(models_dir)

    if root_dir:
        print(f'We have {len(dirs)} model folders!')
        for i, mdir in enumerate(dirs):
            print(f'Test models {1+i}/{len(dirs)}')
            if is_root_dir(mdir):
                print('This was another root dir ', mdir)
                continue
            mdir = models_dir + add_trailing_slash(mdir)
            sdict_files = get_sdict_files(mdir)
            run_experiments(config,
                            models_dir=mdir,
                            sdict_files=sdict_files,
                            verbose=verbose)
    else:
        run_experiments(config, models_dir, sdict_files, verbose=verbose)


def run_experiments(config, models_dir, sdict_files, verbose=False):
    print(f'\nRun experiments for models in {models_dir}')
    print('found files', sdict_files)

    # Torch
    torch.manual_seed = config["misc"].get('random_seed', 1)

    use_cuda = config['misc'].get(
        'use_cuda', True) and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    num_workers = config['misc'].get('num_workers', 1)
    pin_memory = config['misc'].get('pin_memory', True)

    kwargs = {'num_workers': num_workers,
              'pin_memory': pin_memory} if use_cuda else {}
    print(f'device: {device}')

    ts = datetime.now().strftime('%y%m%d-%H%M%S')

    test_name = config['experiments'].get('experiment_name', 'sparse_vs_dense')

    # Metadata

    if 'metadata.json' in sdict_files:
        with open(models_dir + 'metadata.json', 'rb') as fo:
            metadata = json.loads(fo.read())
    else:
        metadata = {}  # there was no metadata

    # Data
    # TODO somehow inform the user if his data settings are different as the ones that the models were trained with

    # option to signal with the config that experiments should be run on the same data
    # as the training
    use_training_setup = config['data'].get('use_training_setup', True)

    data_conf_str = 'training metadata' if use_training_setup else 'configuration file'
    print(f'get data configuration from {data_conf_str}')

    if use_training_setup:
        assert 'data' in metadata, 'we cannot use training setup, because there was no metadata'
        data_config = metadata['data']
    else:
        data_config = config['data']

    assert 'data_dir' in data_config, 'define the data directory'
    data_dir = data_config.get('data_dir')

    data_name = data_config.get('data_name', 'fashion')
    print(f'Use {data_name} dataset from {data_dir}')

    normalize_data = data_config.get('normalize_data', False)
    if normalize_data:
        normalize_by_features = data_config.get('normalize_by_features', True)
        normalize_by_moving = data_config.get('normalize_by_moving', True)
        normalize_by_scaling = data_config.get('normalize_by_scaling', True)
    else:
        normalize_by_features = normalize_by_moving = normalize_by_scaling = False

    batch_size = data_config.get('batch_size', 60)

    # one can override the test batch size of the training setup if wanted
    if 'test_batch_size' in config['data']:
        test_batch_size = config['data']['test_batch_size']
    else:
        test_batch_size = data_config.get('test_batch_size', 1000)

    # document to the experiment config the values we actually used
    config['data'] = {
        "normalize_data": normalize_data,
        "normalize_by_features": normalize_by_features,
        "normalize_by_moving": normalize_by_moving,
        "normalize_by_scaling": normalize_by_scaling,
        "batch_size": batch_size,
        "test_batch_size": test_batch_size,
        "data_name": data_name,
        "data_dir": data_dir
    }

    train_loader, test_loader = get_train_and_test_loaders(
        data_dir, data_name, batch_size, test_batch_size, normalize_data, normalize_by_features, normalize_by_moving, normalize_by_scaling, kwargs)

    train_samples = train_loader.dataset.data.shape[0]
    input_features = train_loader.dataset.data.shape[1] * \
        train_loader.dataset.data.shape[2]
    output_dim = len(train_loader.dataset.classes)

    print(
        f'train data {train_loader.dataset.data.shape} on device {train_loader.dataset.data.device}')
    print(
        f'test data {test_loader.dataset.data.shape} on device {test_loader.dataset.data.device}')

    # Experiments

    scenario_blacklist = config['experiments'].get('scenario_blacklist', [])

    run_all_scenarios = len(scenario_blacklist) == 0

    test_dict = config['experiments'].get('test_dict', {
        "ar_2d": True,
        "trees": True,
        "specialization": True,
        "other_stats": True
    })

    # [0, 1, 9]  # shirts, pants, shoes for fashion
    # None  # to randomize classes
    classes = config['experiments'].get('image_classes', None)

    # how much one class samples should be covered?
    class_coverage = config['experiments'].get('class_coverage', [0.95, 0.99])

    plane_through_origin = config['experiments'].get(
        'plane_through_origin', False)
    use_three_samples = not plane_through_origin

    average_over_images = config['experiments'].get('average_over_images', 20)

    # Mining LPs with Decision Trees
    lp_samples = config['experiments'].get('lp_samples', 5000)

    # Files and saving

    save_results = config['experiments'].get('save_results', True)

    models_str = f'{metadata["misc"].get("notebook_name", "")}_{metadata["models"].get("models_total", "")}-models' if 'models' in metadata else ''
    folder_name = f'{data_name}-{ts}-{test_name}_{models_str}'
    saving_folder = f'results/{folder_name}/'
    if save_results and not os.path.isdir(saving_folder):
        print('Lets create folder for results, ', saving_folder)
        os.mkdir(saving_folder)
    elif not save_results:
        print('Results are NOT saved')

    # Models
    print('\n# Models')

    loss_fn = torch.nn.CrossEntropyLoss()

    sdicts = {}

    for file_name in sdict_files:
        name, ending = file_name.split('.')
        if ending != 'pt':
            continue
        sdicts[name] = torch.load(f'{models_dir}/{file_name}')

    hyperparams = sdicts['hyperparams']

    models_dict = {}
    print('## load models')
    for name in sdicts:
        if 'models' in name:
            print('\t', name)
            models_dict[name.split('_')[-1]] = sdicts[name]

    print('model hyperparams', hyperparams)

    # Init variables

    # extract variables from hyperparams

    param_keys = OrderedDict()
    for name in hyperparams:
        param_keys[name] = [
            hyp[0][-1] for hyp in hyperparams[name]
        ]

    model_architectures = []
    for name in hyperparams:
        model_architectures.append(name)

    print('model_architectures', model_architectures)

    # TODO more flexible for different architectures
    n_densities = len(hyperparams[model_architectures[0]])
    n_sizes = len(hyperparams[model_architectures[0]][-1])

    if save_results:
        with open(saving_folder + 'config.json', 'w') as fo:
            fo.write(json.dumps(config))
        with open(saving_folder + 'metadata.json', 'w') as fo:
            fo.write(json.dumps(metadata))
        torch.save(hyperparams, saving_folder + 'hyperparams.pt')

    ar_data_2D = OrderedDict()
    tree_data = OrderedDict()
    lp_data = OrderedDict()
    specialization_data = OrderedDict()
    uaps = OrderedDict()
    ulps = OrderedDict()
    entropy_stats = OrderedDict()
    purity_stats = OrderedDict()
    horizon_stats = OrderedDict()
    hor_dists = OrderedDict()
    acc_data = OrderedDict()

    data = OrderedDict()
    data['ar_2d'] = ar_data_2D
    data['trees'] = tree_data
    data['lps'] = lp_data
    data['specialization'] = specialization_data
    data['uaps'] = uaps
    data['ulps'] = ulps
    data['entropy_stats'] = entropy_stats
    data['purity_stats'] = purity_stats
    data['horizon_stats'] = horizon_stats
    data['horizon distributions'] = hor_dists
    data['accuracy'] = acc_data

    def save_data():
        if save_results:
            torch.save(data, saving_folder + 'data.pt')
            print('\t---data saved---')

    # Run tests for models

    # ## Activation Regions
    if run_all_scenarios:
        print('Run tests for all scenarions:\n')
    else:
        print('DONT run tests for scenarios: ', scenario_blacklist, '\n')

    for scenario in models_dict:
        if scenario in scenario_blacklist:
            continue

        print(f'\n# Scenario {scenario}')
        models = models_dict[scenario]

        n_models = {}
        for name in models.keys():
            for params in models[name].keys():
                n_models[name] = len(models[name][params][0])
                break
        print(f'n_models: {n_models}')

        models_total = count_models_n_total_from_hyperparams(
            hyperparams, n_models)
        print(f'models in total {models_total}')

        print_model_information(models)

        print('Tests:')

        # Layer patterns

        if test_dict['trees'] or test_dict['specialization']:
            print('\n## Get layer patterns, because we need them later')

            lps_arc, targets_arc, maxpat_arc = get_layer_patterns(models, test_loader,
                                                                  class_coverage,
                                                                  lp_samples, param_keys,
                                                                  n_models, models_total,
                                                                  input_features=28**2, verbose=True)

            lp_data[scenario] = (lps_arc, targets_arc)

            print('')

            save_data()

        # 2D ARs & Specialization

        if test_dict['ar_2d'] and test_dict['specialization']:
            print('\n## Compute 2D ARs AND specialization data in one go')

            ar_stats, spec_arc = compute_local_2D_ARs_and_specialization_for_models(models,
                                                                                    use_three_samples=use_three_samples,
                                                                                    data_loader=test_loader,
                                                                                    classes=classes,
                                                                                    maxpat_arc=maxpat_arc,
                                                                                    average_over_images=average_over_images,
                                                                                    param_keys=param_keys,
                                                                                    n_models=n_models,
                                                                                    hyperparams=hyperparams,
                                                                                    models_total=models_total,
                                                                                    verbose=True)
            specialization_data[scenario] = spec_arc
            ar_data_2D[scenario] = ar_stats

        elif test_dict['ar_2d']:
            print('\n## 2D activation regions, no specialization data')

            ar_data_2D[scenario] = compute_local_2D_ARs_for_models(models,
                                                                   use_three_samples=use_three_samples,
                                                                   data_loader=test_loader,
                                                                   classes=classes,
                                                                   average_over_images=average_over_images,
                                                                   param_keys=param_keys,
                                                                   n_models=n_models,
                                                                   hyperparams_for_sparse=hyperparams,
                                                                   models_total=models_total,
                                                                   verbose=True)
        elif test_dict['specialization']:
            print('\n## Specialization data, no 2D ARs')
            lps_arc, targets_arc, spec_arc = get_layer_patterns_and_specialization_data(models,
                                                                                        test_loader,
                                                                                        lp_samples=lp_samples,
                                                                                        param_keys=param_keys,
                                                                                        use_three_samples=use_three_samples,
                                                                                        n_models=n_models,
                                                                                        models_total=models_total,
                                                                                        average_over_images=average_over_images,
                                                                                        coverage=class_coverage,
                                                                                        classes=classes,
                                                                                        verbose=True)
            lp_data[scenario] = (lps_arc, targets_arc)
            specialization_data[scenario] = spec_arc
        else:
            print('\ndont compute 2D ARs nor specialization metrics')

        save_data()
        # Trees

        if test_dict['trees']:
            print(f'\n## Teach a decision tree for each models layer patterns')

            tree_data_arc = OrderedDict()
            for sparse_from in model_architectures:
                layer_i = -2  # the last hidden layer is -2
                tree_d = tree_data_for_layer_i_constant_param(
                    lps_arc, targets_arc, sparse_from, layer_i, param_keys)
                tree_data_arc[sparse_from] = tree_d

            tree_data[scenario] = tree_data_arc
            print()
            save_data()
        else:
            print('\n dont compute tree data')

        # Other stats

        if test_dict['other_stats']:
            print('## Other statistics')

            stats_ini = unique_patterns_and_stats_for_models(models,
                                                             test_loader,
                                                             n_sizes,
                                                             n_models,
                                                             verbose=verbose,
                                                             input_features=input_features)

            uaps[scenario] = stats_ini[0]
            ulps[scenario] = stats_ini[1]
            entropy_stats[scenario] = stats_ini[2]
            purity_stats[scenario] = stats_ini[3]
            horizon_stats[scenario] = stats_ini[4]
            hor_dists[scenario] = stats_ini[5]

            save_data()
        else:
            print('\ndont compute other statistics')

        print('\n## Accuracies')

        accuracies = compute_acc_for_trained(models, test_loader, loss_fn)

        acc_data[scenario] = accuracies

        save_data()

        print(f'Scenario {scenario} finished!\t\t\t\t\t')


if __name__ == '__main__':
    configs, verbose = parse_arguments()

    n_conf = len(configs)
    for i, config in enumerate(configs):
        print(f'Run tests defined in config {config}, {i+1}/{n_conf}')
        run_experiments_models(configs[config], verbose=verbose)
