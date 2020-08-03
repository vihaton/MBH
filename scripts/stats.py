from collections import OrderedDict

from scipy.stats import entropy
import numpy as np
import torch

from .activation_regions import *
from .activation_patterns import *


def aggregate_from_aps_and_labels(aps, labels):
    '''labels can be either predictions or true labels'''
    aps_dict = {}

    for i, ap in enumerate(aps):
        ap_str = activation_pattern_to_str(ap)
        if ap_str not in aps_dict:
            preds_agg = np.zeros((14,))
        else:
            preds_agg = aps_dict[ap_str]

        preds_agg[labels[i]] += 1
        aps_dict[ap_str] = preds_agg

    aggregate = np.array(list(aps_dict.values()))

    # sum is the 11th from the left
    aggregate[:, 10] = np.sum(aggregate, axis=1)
    # entropy for every row (without the sum)
    aggregate[:, 11] = entropy(aggregate[:, :10], axis=1)
    # sum x entropy, aka weighted entropy
    aggregate[:, 12] = aggregate[:, 10] * aggregate[:, 11]
    # how many different classes per ap, aka purity
    aggregate[:, 13] = np.sum(aggregate[:, :10] != 0, axis=1)

    return aggregate


def entropy_from_aggregate(aggregate, average=False):
    '''
    aggregate shape [#unique aps, 14]
    returns table of averages over entropies and weighted entropies, ie. len=2
    '''
    if average:
        return np.average(aggregate, axis=0)[-3:-1]
    else:
        return np.sum(aggregate, axis=0)[-3:-1]


def purity_from_aggregate(aggregate):
    '''
    how many APs, where there are exactly #i classes?
    '''
    purities = np.sum(aggregate[:, :10] != 0, axis=1)
    n_classes, counts = np.unique(purities, return_counts=True)

    purity_agg = np.zeros(10)
    for ind in range(len(n_classes)):
        i = n_classes[ind]
        count = counts[ind]
        purity_agg[i-1] = count

    return purity_agg


def count_data_over_the_horizons(layer_patterns):
    '''
    Counts the number of horizons the datapoint was over
    layer patterns shae [#patterns, #layers][#neurons in layer]
    returns number of horizons (ave, min, max), horizons by neurons
    '''
    layer_1 = np.array(layer_patterns[:, 0])
    n_neurons = layer_1[0].size
    l1 = np.concatenate(layer_1).reshape(-1, n_neurons)
    assert ((np.sum(l1[0] == layer_1[0])) == len(layer_1[0])
            ), "we didn't concatenate + reshape right"

    # for each neuron, how many images crossed it's horizon?
    horizons_by_neurons = np.sum(l1, axis=0)
    over_horizons = np.sum(l1, axis=1)  # sum all the neurons for each image

    ave, mi, ma = np.average(over_horizons), np.min(
        over_horizons), np.max(over_horizons)

    return (ave, mi, ma), horizons_by_neurons


def unique_patterns_and_stats_for_models(models, test_loader, n_sizes, n_models, verbose=True, input_features=28*28):
    '''
    returns
        lvl 0: tuple of OrderedDicts (uaps, ulps, stats, purity, horizon) key:model name
            1: both have lists for pruning rates
                2: that store the number of unique patterns in lists
    '''
    prgrss_architectures = 100 / len(models.keys())

    uaps_model, ulps_model = OrderedDict(), OrderedDict()  # lvl 0
    stats_model = OrderedDict()
    purity_model = OrderedDict()
    horizon_model = OrderedDict()
    hor_dist_model = OrderedDict()
    for i, name in enumerate(models):
        if verbose:
            print(f'# get patterns for sparse {name}s')
        prgrss_model = prgrss_architectures / \
            n_models[name] / (n_sizes + 1) / n_sizes * 2

        k = 0
        uaps_pr, ulps_pr, stats_pr, purity_pr, horizon_pr, hor_dist_pr = [
        ], [], [], [], [], []  # lvl 1
        for params in models[name]:
            uaps_h, ulps_h, stats_h, purity_h, horizon_h, hor_dist_h = [
            ], [], [], [], [], []  # lvl 2
            if verbose:
                print(f'## parameters {params}')

            for j, models_h in enumerate(models[name][params]):
                if verbose:
                    print(f'### hyperparam setup {j}')
                unique_aps, unique_lps, stats, purities, horizon, hor_dist = [
                ], [], [], [], [], []  # lvl 3

                for n, model in enumerate(models_h):
                    k += 1
                    print(
                        f'progress {round(i*prgrss_architectures+k*prgrss_model,2)}%', end='\r')

                    # record the APs while feeding test data to the network
                    ape = get_APs(model, test_loader,
                                  input_features=input_features)

                    # extract the data from observing the APs earlier
                    aps, lps, preds, targets = ape.get_patterns()

                    # entropy of preds
                    aggregate_preds = aggregate_from_aps_and_labels(aps, preds)
                    pred_stats = entropy_from_aggregate(aggregate_preds)

                    # entropy of targets
                    aggregate_targets = aggregate_from_aps_and_labels(
                        aps, targets)
                    target_stats = entropy_from_aggregate(aggregate_targets)

                    stats.append((pred_stats, target_stats))

                    # purity for targets
                    purity_targets = purity_from_aggregate(aggregate_targets)
                    purities.append(purity_targets)

                    # how does the data cross the first layer neuron horizons?
                    hor_stats, horizons_by_neurons = count_data_over_the_horizons(
                        np.array(lps))
                    horizon.append(hor_stats)
                    hor_dist.append(horizons_by_neurons)

                    # unique activation and layer patterns
                    uap, ulp = calculate_unique_patterns(ape, name, verbose)
                    unique_aps.append(uap)
                    unique_lps.append(ulp)

                uaps_h.append(unique_aps)
                ulps_h.append(unique_lps)
                stats_h.append(stats)
                purity_h.append(purities)
                horizon_h.append(horizon)
                hor_dist_h.append(hor_dist)

            uaps_pr.append(uaps_h)
            ulps_pr.append(ulps_h)
            stats_pr.append(stats_h)
            purity_pr.append(purity_h)
            horizon_pr.append(horizon_h)
            hor_dist_pr.append(hor_dist_h)
            if verbose:
                print()

        uaps_model[name] = np.array(uaps_pr)
        ulps_model[name] = np.array(ulps_pr)
        stats_model[name] = np.array(stats_pr)
        purity_model[name] = np.array(purity_pr)
        horizon_model[name] = np.array(horizon_pr)
        hor_dist_model[name] = np.array(hor_dist_pr)
        if verbose:
            print()

    return uaps_model, ulps_model, stats_model, purity_model, horizon_model, hor_dist_model


def unique_patterns_and_stats_for_sparse_models(sparse_models, test_loader, one_shot_pruning_rates, n_models, return_horizons=False, verbose=True, input_features=28*28):
    '''
    returns
        lvl 0: tuple of OrderedDicts (uaps, ulps, stats, purity, horizon) key:model name
            1: both have lists for pruning rates
                2: that store the number of unique patterns in lists
    '''
    prgrss_architectures = 100 / len(sparse_models.keys())
    prgrss_prs = prgrss_architectures / len(one_shot_pruning_rates)

    uaps_model, ulps_model = OrderedDict(), OrderedDict()  # lvl 0
    stats_model = OrderedDict()
    purity_model = OrderedDict()
    horizon_model = OrderedDict()
    hor_dist_model = OrderedDict()
    for i, name in enumerate(sparse_models):
        if verbose:
            print(f'# get patterns for sparse {name}s')
        sms = sparse_models[name]
        prgrss_model = prgrss_prs / n_models[name]

        uaps_pr, ulps_pr, stats_pr, purity_pr, horizon_pr, hor_dist_pr = [
        ], [], [], [], [], []  # lvl 1
        for k, pr in enumerate(one_shot_pruning_rates):
            unique_aps, unique_lps, stats, purities, horizon, hor_dist = [
            ], [], [], [], [], []  # lvl 2
            if verbose:
                print(f'## pruning rate {pr}')

            for j, smodel in enumerate(sms[pr]):
                print(
                    f'progress {round(i*prgrss_architectures+k*prgrss_prs+j*prgrss_model,2)}%', end='\r')

                # record the APs while feeding test data to the network
                ape = get_APs(smodel, test_loader,
                              input_features=input_features)

                # extract the data from observing the APs earlier
                aps, lps, preds, targets = ape.get_patterns()

                # entropy of preds
                aggregate_preds = aggregate_from_aps_and_labels(aps, preds)
                pred_stats = entropy_from_aggregate(aggregate_preds)

                # entropy of targets
                aggregate_targets = aggregate_from_aps_and_labels(aps, targets)
                target_stats = entropy_from_aggregate(aggregate_targets)

                stats.append((pred_stats, target_stats))

                # purity for targets
                purity_targets = purity_from_aggregate(aggregate_targets)
                purities.append(purity_targets)

                # how does the data cross the first layer neuron horizons?
                hor_stats, horizons_by_neurons = count_data_over_the_horizons(
                    np.array(lps))
                horizon.append(hor_stats)
                hor_dist.append(horizons_by_neurons)

                # unique activation and layer patterns
                uap, ulp = calculate_unique_patterns(ape, name, verbose)
                unique_aps.append(uap)
                unique_lps.append(ulp)

            uaps_pr.append(unique_aps)
            ulps_pr.append(unique_lps)
            stats_pr.append(stats)
            purity_pr.append(purities)
            horizon_pr.append(horizon)
            hor_dist_pr.append(hor_dist)

        uaps_model[name] = np.array(uaps_pr)
        ulps_model[name] = np.array(ulps_pr)
        stats_model[name] = np.array(stats_pr)
        purity_model[name] = np.array(purity_pr)
        horizon_model[name] = np.array(horizon_pr)
        hor_dist_model[name] = np.array(hor_dist_pr)
        if verbose:
            print()

    if return_horizons:
        return uaps_model, ulps_model, stats_model, purity_model, horizon_model, hor_dist_model
    else:
        return uaps_model, ulps_model, stats_model, purity_model


def compute_weight_and_bias_differences(original_layers, trained_layers):
    '''
    weight wise difference
        mean(trained - original)
    and bias wise differense
        trained - original
    for every layer, returns weight and bias differences
        [w/b, #hidden_neurons]
    '''

    diff_w, diff_b = [], []
    for d, layer in enumerate(original_layers):
        w_org = layer.weight.detach().numpy()
        w_trained = trained_layers[d].weight.detach().numpy()
        b_org = layer.bias.detach().numpy()
        b_trained = trained_layers[d].bias.detach().numpy()

        w_diff = np.mean(w_trained - w_org, axis=1)
        b_diff = b_trained - b_org

        diff_w.append(w_diff)
        diff_b.append(b_diff)
    return np.array((np.concatenate(diff_w), np.concatenate(diff_b)))


def aggregate_wb_differences(models_org: list, models_trained: list, return_by_layers=False):
    '''
    return [#models, weights/biases, #hidden_neurons] calculated from the models

    if return_by_layers == True, then return also
        [#layers, mean over models, w/b, #hidden_neurons_of_that_layer]
    '''
    differences = []  # [#models, weight/bias, #layers, hidden_dim]
    for n, model_org in enumerate(models_org):
        layers_trained = list(models_trained[n].children())
        layers_original = list(model_org.children())
        diff = compute_weight_and_bias_differences(
            layers_original, layers_trained)
        differences.append(diff)

    differences = np.array(differences)
    dims = [layer.out_features for layer in layers_original]

    agg = differences.view()

    if return_by_layers:
        by_layers = []
        low_bound = 0
        for dim in dims:
            up_b = low_bound+dim
            by_layers.append(agg[:, :, low_bound:up_b])
            low_bound = up_b
        return by_layers
    else:
        return agg


def compute_weight_differences_for_denses(weight_diffs, dense_models_org, dense_models_trained, by_layers=False):
    for i, name in enumerate(dense_models_org):
        dmodels_org = dense_models_org[name]
        dmodels_trained = dense_models_trained[name]

        aggregates = aggregate_wb_differences(
            dmodels_org, dmodels_trained, by_layers)
        weight_diffs[name] = aggregates
    return weight_diffs


def compute_weight_differences_for_sparses(weight_diffs, scenario, sparse_models_org, sparse_models_trained, by_layers=False):
    for i, name in enumerate(sparse_models_org):
        aggs_by_pr = OrderedDict()
        for j, pr in enumerate(sparse_models_org[name]):
            smodels_org = sparse_models_org[name][pr]
            smodels_trained = sparse_models_trained[name][pr]

            aggregates = aggregate_wb_differences(
                smodels_org, smodels_trained, by_layers)
            aggs_by_pr[pr] = aggregates
        weight_diffs[f'{name}-{scenario}'] = aggs_by_pr
    return weight_diffs


def dark_mask_differences(dark_masks_trained):
    '''
    returns OD: case -> OD: name -> 
            [#ms, (under limit @ init, under limit after training, how many were the same)]
    '''
    dm_diff = OrderedDict()
    cases = ['rnd', 'orig']
    for case in cases:
        dm_diff_arch = OrderedDict()
        dm_ini, dm_trained = dark_masks_trained[f'{case} @ init'], dark_masks_trained[f'{case} trained']
        for name in dm_ini:
            dm_diff_ms = []
            for i in range(dm_ini[name].shape[0]):
                under_limit_init = dm_ini[name][i] == 0
                under_limit_trained = dm_trained[name][i] == 0
                ul_same = np.bitwise_and(under_limit_init, under_limit_trained)
                dm_diff_ms.append([np.sum(under_limit_init), np.sum(
                    under_limit_trained), np.sum(ul_same)])
            dm_diff_arch[name] = np.array(dm_diff_ms)
        dm_diff[case] = dm_diff_arch
    return dm_diff


# Specialization


def get_layer_patterns_and_specialization_data(models, data_loader,
                                               lp_samples, param_keys,
                                               use_three_samples,
                                               n_models, models_total,
                                               average_over_images,
                                               coverage=1.0, classes=None,
                                               input_features=28**2, verbose=False):
    prgrss_model = 100 / models_total

    n = 0
    lps_arc, targets_arc, spec_arc = OrderedDict(), OrderedDict(), OrderedDict()
    for i, name in enumerate(models):
        if verbose:
            print(f'# get patterns for sparse {name}s')

        lps_param, targets_param, mp_param = [], [], []
        # for each number of parameters,
        for j, params in enumerate(param_keys[name]):
            if verbose:
                print(f'## params {params}\t')

            lps_h, targets_h, mp_h = [], [], []
            # for each hyperparam setup
            for k in range(len(models[name][params])):

                lps_ar, targets_ar, mp_ar = [], [], []
                for model in models[name][params][k]:  # for each model
                    n += 1
                    print(f'progress {round(n*prgrss_model,2)}%', end='\r')

                    # record the APs while feeding test data to the network
                    ape = get_APs(model, data_loader, only_first=False,
                                  lp_samples=lp_samples,
                                  input_features=input_features)

                    # extract the data from observing the APs earlier
                    aps, lps, preds, targets = ape.get_patterns()

                    lps_ar.append(lps)
                    targets_ar.append(targets)

                    # specialization data
                    lpi = parse_layers_patterns_to_numpy(
                        lps, -2)  # last hidden layer aka -2
                    mp_ar.append(specialization_data_for_model(model,
                                                               data_loader,
                                                               lpi,
                                                               targets,
                                                               average_over_images,
                                                               use_three_samples,
                                                               coverage=coverage,
                                                               classes=classes))

                lps_h.append(lps_ar)
                targets_h.append(targets_ar)
                mp_h.append(np.array(mp_ar))

            lps_param.append(lps_h)
            targets_param.append(targets_h)
            mp_param.append(np.array(mp_h))

        lps_arc[name] = np.array(lps_param)
        targets_arc[name] = np.array(targets_param)
        spec_arc[name] = np.array(mp_param)

    return lps_arc, targets_arc, spec_arc


def specialization_data_for_model(model, data_loader, lpi, targets,
                                  average_over_images, use_three_samples,
                                  coverage=1.0,
                                  classes=None):
    n_classes = len(data_loader.dataset.classes)
    random_classes = classes is None

    max_patterns = get_max_patterns_for_classes(
        lpi, targets, n_classes, coverage=coverage)

    spec_data_imgs = []
    for i in range(average_over_images):
        if random_classes:
            classes = get_random_classes(3, n_classes)

        example_imgs = get_example_images(data_loader, classes)

        subspace = span_subspace_with_three_images(example_imgs,
                                                   use_three_samples=use_three_samples)

        subspace.split_by_model(model)

        spec_data_class = []
        for c in range(n_classes):
            neuron_indx, neuron_activations = max_patterns[c]

            if np.sum(neuron_indx) == -1 or len(neuron_indx) == 0:  # no pattern for this class
                area_c = subspace.local_area
            else:
                layer_i = -1  # -1 bc we have recorded neurons only to the last hidden layer
                area_c = subspace.local_area_of_regions_that_satisfy_subpattern(layer_i,
                                                                                neuron_indx,
                                                                                neuron_activations)
            spec_data_class.append(area_c)

        local_area = subspace.local_area
        total_area_classes = np.sum(spec_data_class)
        specialization_ratio = (local_area + 1) / (total_area_classes + 1)

        spec_data_imgs.append(np.array(
            [specialization_ratio, total_area_classes, local_area] + spec_data_class
        ))

    return np.array([
        np.mean(spec_data_imgs, axis=0),
        np.min(spec_data_imgs, axis=0),
        np.max(spec_data_imgs, axis=0)
    ]).T


def specialization_for_blanket(blanket_hv, subspace_hv):
    return 1 - blanket_hv / subspace_hv


def aggregate_specialization_over_several_phenomena(blanket_hvs, space_hv: float):
    assert type(
        blanket_hvs) is np.ndarray, 'blanket hypervolumes should be in a numpy array'
    specializations = np.ones_like(blanket_hvs)
    return np.mean(specializations - (blanket_hvs / space_hv))


def specialization_data_of_subspace(subspace, max_patterns, n_classes):
    '''returns specialization data, shape=[13, #coverages, ave/min/max]'''
    plane_area = subspace.local_area

    spec_data_coverage = []
    for max_patterns_cov in max_patterns:
        spec_data_class, area_data_class = [], []
        # there is maximum pattern for each class

        for c in range(len(max_patterns_cov)):
            neuron_indx, neuron_activations = max_patterns_cov[c]

            if np.sum(neuron_indx) == -1 or len(neuron_indx) == 0:  # no pattern for this class
                area_c = plane_area
            else:
                layer_i = -1  # -1 bc we have recorded neurons only to the last hidden layer

                # this can be time expensive, much less than the splitting though
                area_c = subspace.local_area_of_regions_that_satisfy_subpattern(layer_i,
                                                                                neuron_indx,
                                                                                neuron_activations)

            spec_c = specialization_for_blanket(area_c, plane_area)

            area_data_class.append(area_c)
            spec_data_class.append(spec_c)

        total_area_classes = np.sum(area_data_class)
        spec_ave_classes = np.mean(spec_data_class)

        specialization_ratio = spec_ave_classes

        spec_data_coverage.append(
            [specialization_ratio, total_area_classes, plane_area] + area_data_class + spec_data_class)

    return np.array(spec_data_coverage)


def compute_local_2D_ARs_and_specialization(model, subspace, max_patterns):
    # the time expensive operation
    subspace = subspace.split_by_model(model)

    # stats for 2D ARs
    plane_area = subspace.local_area
    n_r = subspace.number_of_regions()
    ar_stats = (n_r / plane_area, n_r, plane_area)

    n_classes = len(max_patterns[0])  # hack: each class has its max pattern
    spec_data_coverage = specialization_data_of_subspace(
        subspace, max_patterns, n_classes)

    return ar_stats, spec_data_coverage


def get_n_spanning_image_groups(n, data_loader, classes):
    '''returns images, their classes'''
    n_classes = len(data_loader.dataset.classes)
    random_classes = classes is None

    images, labels = [], []
    for i in range(n):
        if random_classes:
            classes = get_random_classes(3, n_classes)

        images.append(get_example_images(data_loader, classes))
        labels.append(classes)

    return images, labels


def compute_2D_ARs_and_specialization_for_subspaces(model, example_image_sets, average_over_images,
                                                    use_three_samples, max_patterns):
    img_i = 0
    ars_img, spec_img = [], []
    for i, example_imgs in enumerate(example_image_sets):
        img_i += 1
        end = '\n' if img_i == len(example_image_sets) else '\r'
        print(
            f'computing stats for subspace {img_i}/{average_over_images}', end=end)

        subspace = span_subspace_with_three_images(
            example_imgs, use_three_samples=use_three_samples)

        ars_m, spec_m = compute_local_2D_ARs_and_specialization(
            model, subspace, max_patterns)

        ars_img.append(ars_m)
        spec_img.append(spec_m)

    ars_img = np.array(ars_img)
    spec_img = np.array(spec_img)

    return ars_img, spec_img


def compute_local_2D_ARs_and_specialization_for_models(models, use_three_samples,
                                                       data_loader, classes,
                                                       maxpat_arc,
                                                       average_over_images, param_keys,
                                                       n_models, hyperparams,
                                                       models_total, verbose=False):
    ar_stats, spec_arc = OrderedDict(), OrderedDict()

    # use the same three images for all the models
    n_example_sets, _ = get_n_spanning_image_groups(
        average_over_images, data_loader, classes)

    for name in models:
        m_i = 0
        ars_param, spec_param = [], []
        for j, params in enumerate(param_keys[name]):
            ars_j, spec_j = [], []
            for i, models_n in enumerate(models[name][params]):
                ars, spec_ar = [], []
                for n, model in enumerate(models_n):
                    m_i += 1
                    if verbose:
                        print(
                            f'computing stats for model {m_i}/{models_total}')

                    maxpat_m = maxpat_arc[name][j][i][n]

                    ars_m, spec_m = compute_2D_ARs_and_specialization_for_subspaces(model,
                                                                                    n_example_sets,
                                                                                    average_over_images,
                                                                                    use_three_samples,
                                                                                    maxpat_m)

                    ars.append(ars_m)
                    spec_ar.append(spec_m)

                ars_j.append(ars)
                spec_j.append(spec_ar)

            ars_param.append(np.array(ars_j))
            spec_param.append(np.array(spec_j))

        ar_stats[name] = np.array(ars_param)
        spec_arc[name] = np.array(spec_param)

    return ar_stats, spec_arc
