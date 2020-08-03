from torch.nn.utils.prune import L1Unstructured, RandomUnstructured
from copy import deepcopy

import numpy as np
import torch
from collections import OrderedDict
from torch.nn.utils.prune import RandomStructured, LnStructured, identity


def init_for_pruning(smodel, to_be_pruned: [str], last_layer_index=-1):

    # don't prune the output layer
    for layer in list(smodel.children())[:last_layer_index]:
        for pname in to_be_pruned:
            identity(layer, pname)


def calculate_mask_sparsity(model):
    '''
    returns a tuple (ms for the whole model, [ms by layers])'''

    params_l, remain_l = [], []
    for layer in model.children():
        mask = layer.mask
        n_params = mask.shape[0] * mask.shape[1]
        params_l.append(n_params)
        remain_l.append(np.sum(mask.numpy()))
    ms = np.sum(remain_l) / np.sum(params_l)
    ms_l = np.array(remain_l) / np.array(params_l)
    return ms, ms_l


def infer_one_shot_prs(sparse_models):
    return list(sparse_models['lenet'].keys())

# pruning with pytorch

# PRE-DEFINED SPARSITY


def prune_with_predefined_sparsity(model, density, random_mask=True, prune_all_layers=False, unstructured=True):
    amount = 1-density
    last_layer_index = len(list(model.children())) if prune_all_layers else -1

    masks = pruning_masks(model,
                          unstructured=unstructured,
                          amount=amount,
                          random_mask=random_mask,
                          style='nodes',
                          last_layer_index=last_layer_index)

    init_for_pruning(model, ['weight'],
                     last_layer_index=last_layer_index)
    for j, layer in enumerate(list(model.children())[:last_layer_index]):
        del layer._buffers['weight_mask']
        layer.register_buffer('weight_mask', masks[j])
    return model


def pruning_masks(dmodel, unstructured: bool, amount: float, random_mask=False, n_norm=1, style='nodes', last_layer_index=-1):
    '''
    get pruning masks for weights (not biases) in structured manner,
    the same amount on every hidden layer (i.e. the output layer is excluded!)
    returns mask to be applied

    '''
    masks = []

    # decide the pruning method 1) unstructured/structured and 2) random/magnitude
    if unstructured:
        pruning_method = RandomUnstructured(
            amount=amount) if random_mask else L1Unstructured(amount=amount)
    else:  # structured pruning, should we prune nodes or channels?
        dim = 0
        if style == 'channels':
            dim = -1

        pruning_method = RandomStructured(amount=amount, dim=dim) if random_mask else LnStructured(
            amount=amount, n=n_norm, dim=dim)

    for layer in list(dmodel.children())[:last_layer_index]:
        trained_weight = getattr(layer, 'weight')
        masks.append(pruning_method.compute_mask(
            trained_weight, torch.ones_like(trained_weight)))

    return masks


def prune_models(models_org, models_trained, unstructured: bool, amount: float, random_mask: bool, random_init: bool, prune_all_layers: bool):
    '''
    models is list of dense models for one architecture
    '''

    sms = []
    for i, model in enumerate(models_trained):
        # create the sparse model
        sparse_m = deepcopy(models_org[i])

        last_layer_index = len(list(sparse_m.children())
                               ) if prune_all_layers else -1

        # get the pruning masks
        dmodel = deepcopy(model)
        masks = pruning_masks(dmodel,
                              unstructured=unstructured,
                              amount=amount,
                              random_mask=random_mask,
                              style='nodes',
                              last_layer_index=last_layer_index)

        init_for_pruning(sparse_m, ['weight'],
                         last_layer_index=last_layer_index)
        for j, layer in enumerate(list(sparse_m.children())[:last_layer_index]):
            del layer._buffers['weight_mask']
            layer.register_buffer('weight_mask', masks[j])

        # if random init, re-initialize the params
        if random_init:
            sparse_m.init_params()

        sms.append(sparse_m)
    return sms

# PRUNE NETWORKS


def create_random_init_models(dense_models_org, dense_models, one_shot_pruning_rates,
                              layer_wise_pruning, prune_weights,
                              n_models, input_features, output_dim,
                              random_init=True, random_mask=True,
                              names=['lenet', 'deepfc'],
                              prune_all_layers=False):

    random_init_models = OrderedDict()
    for name in names:
        rim = OrderedDict()
        for pr in one_shot_pruning_rates:
            print(f'{name}: pr {pr}', end='\r')
            rim_pr = prune_models(models_org=dense_models_org[name],
                                  models_trained=dense_models[name],
                                  unstructured=prune_weights,
                                  amount=1-pr,
                                  random_mask=random_mask, random_init=random_init,
                                  prune_all_layers=prune_all_layers)
            rim[pr] = rim_pr

        random_init_models[name] = rim
    print("", end='\r')
    return random_init_models


def create_winning_tickets(dense_models_org, dense_models_trained, prune_weights,
                           one_shot_pruning_rates, layer_wise_pruning,
                           n_models, input_features, output_dim, names=[
                               'lenet', 'deepfc'],
                           prune_all_layers=False):
    original_init_models = OrderedDict()
    for name in names:
        oim = OrderedDict()
        for pr in one_shot_pruning_rates:
            oim_pr = prune_models(dense_models_org[name], dense_models_trained[name],
                                  unstructured=prune_weights, amount=1-pr,
                                  random_mask=False, random_init=False,
                                  prune_all_layers=prune_all_layers)
            oim[pr] = oim_pr

        original_init_models[name] = oim
    return original_init_models


# OLD NODE WISE PRUNING

def get_pruned_dim(pruning_rate, original_dim, round=False):
    r = 0.5 if round else 0
    dim = int(r + pruning_rate * original_dim)
    return dim if dim > 0 else 1


def get_dim(pruning_rate, layer):
    return get_pruned_dim(pruning_rate, layer.out_features)


def original_weights_for_pruned_lenet(original_weights, pruned_nodes_idx):
    l1 = original_weights[0][pruned_nodes_idx[0], :]
    l2 = original_weights[1][pruned_nodes_idx[1], :][:, pruned_nodes_idx[0]]
    out = original_weights[2][:, pruned_nodes_idx[1]]
    return [l1, l2, out]


def apply_node_wise_pruning_to_weights(original_weights, pruned_nodes_idx):
    pruned = []
    n_layers = len(original_weights)
    for n, w in enumerate(original_weights):
        if n == 0:  # input layer
            pruned.append(w[pruned_nodes_idx[n], :])
        elif n == n_layers - 1:  # output layer
            pruned.append(w[:, pruned_nodes_idx[n-1]])
        else:  # hidden layers
            pruned.append(w[pruned_nodes_idx[n], :][:, pruned_nodes_idx[n-1]])

    return pruned


def prune_nodes_large_final(trained_weights, pruning_rate: float):
    '''
    trained weights is a list of numpy arrays
    pruning rate e [0,1]
    '''
    target_dims = [int(trained_weights[n].shape[0] * (1-pruning_rate))
                   for n in range(len(trained_weights))]
    pruned_nodes_idx = []
    # don't prune the output layer
    for i, w in enumerate(trained_weights[:-1]):
        w_mean = np.mean(np.abs(w), 1)
        idx = np.argsort(w_mean, )
        max_nodes_idx = idx[-target_dims[i]-1:-1]
        pruned_nodes_idx.append(max_nodes_idx)
    return pruned_nodes_idx


def original_weights_for_pruned_deepfc(original_weights, pruned_nodes_idx):
    l1 = original_weights[0][pruned_nodes_idx[0], :]
    l2 = original_weights[1][pruned_nodes_idx[1], :][:, pruned_nodes_idx[0]]
    l3 = original_weights[2][pruned_nodes_idx[2], :][:, pruned_nodes_idx[1]]
    out = original_weights[3][:, pruned_nodes_idx[2]]
    return [l1, l2, l3, out]

# WEIGHT WISE PRUNING


def compute_pruning_mask_large_final(abs_weights, pruning_perc):
    '''abs_weights is a flat tensor of absolute values of weights'''
    threshold = np.percentile(abs_weights, pruning_perc)
    mask = abs_weights > threshold
    return mask


def prune_weights_large_final(model, pruning_perc: float, layer_wise_pruning: bool):
    '''
    pruning_perc e [0,100]
    returns mask or list of masks that represents the pruning
    '''
    network_weights = []
    dims = []

    for child in model.children():
        param = child.weight
        dims.append(param.data.shape)
        layer_weights = param.data.abs().numpy().flatten()
        network_weights.append(layer_weights)

    if layer_wise_pruning:
        masks = [
            compute_pruning_mask_large_final(layer_weights, pruning_perc).reshape(dims[i]) for i, layer_weights in enumerate(network_weights)
        ]
    else:
        masks = []
        mask = compute_pruning_mask_large_final(
            np.concatenate(network_weights), pruning_perc)
        n_layer_params = [weights.shape[0] for weights in network_weights]
        ind = 0
        for i, n_params in enumerate(n_layer_params):
            masks.append(mask[ind:ind+n_params].reshape(dims[i]))
            ind += n_params

    return masks


def get_random_masks(model, mask_sparsity):
    masks = []
    for layer in model.children():
        param = layer.weight.detach().numpy()
        n_weights = param.shape[0] * param.shape[1]
        mask = np.zeros(n_weights)

        weights_left = get_pruned_dim(mask_sparsity, n_weights)
        keep_idx = np.random.choice(
            n_weights, size=weights_left, replace=False)

        mask[keep_idx] = 1
        mask = mask.reshape(param.shape)

        assert np.sum(mask) == weights_left

        masks.append(mask)

    return masks


# Sanity checks

def get_named_params(model, which="weight"):
    params = []
    for child in model.children():
        for pname, param in child.named_parameters():
            if which in pname:
                params.append(param)
    return params


def list_of_tensors_to_numpy(tensors):
    return [tensors[n].detach().numpy() for n in range(len(tensors))]


hist_bins = np.array([-0.725, -0.675, -0.625, -0.575, -0.525, -0.475, -0.425, -0.375, -0.325, -0.275, -0.225, -0.175, -0.125, -
                      0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725])


def check_similarity(array1, array2, bins=hist_bins):
    if not isinstance(array1, np.ndarray):
        array1 = array1.detach().numpy()
    if not isinstance(array2, np.ndarray):
        array2 = array2.detach().numpy()

    assert (isinstance(array1, np.ndarray) and isinstance(
        array2, np.ndarray)), 'inputs should be numpy arrays!'

    same_mask = array1 == array2
    same = np.sum(same_mask)
    hist_same_values, bin_edges = np.histogram(array1[same_mask], bins=bins)
    total = array1.size
    return (same, total), hist_same_values


def check_param_similarity_to_dense(sparse_models, dense_models, n_models, one_shot_pruning_rates, layer_wise_pruning=True, dense_models_trained=None, verbose=True, which='weight'):
    res, histograms = OrderedDict(), OrderedDict()
    for sparse_from in sparse_models:
        res_pr, hist_pr = [], []
        for pr in one_shot_pruning_rates:
            res_m, hist_m = [], []
            for i in range(n_models[sparse_from]):
                sparse_w = get_named_params(
                    sparse_models[sparse_from][pr][i], which=which)
                sparse_w = list_of_tensors_to_numpy(sparse_w)
                dense_w = get_named_params(
                    dense_models[sparse_from][i], which=which)
                dense_w = list_of_tensors_to_numpy(dense_w)

                if sparse_w[0].shape != dense_w[0].shape:
                    # the networks are different size, we need to prune the dense to the size of sparse
                    # we need the trained models if we need to prune!
                    assert dense_models_trained is not None

                    dense_trained_w = list_of_tensors_to_numpy(
                        get_named_params(
                            dense_models_trained[sparse_from][i], which=which)
                    )

                    pruned_nodes_idx = prune_nodes_large_final(
                        dense_trained_w, pruning_rate=1-pr)
                    dense_w = apply_node_wise_pruning_to_weights(
                        dense_w, pruned_nodes_idx)

                if verbose:
                    print(f'{sparse_from} mask sparsity {pr}, model {i}')
                res_l, hist_l = [], []
                for j in range(len(sparse_w)):
                    the_same, hist = check_similarity(
                        sparse_w[j], dense_w[j], bins=hist_bins)
                    res_l.append(the_same)
                    hist_l.append(hist)
                    if verbose:
                        print(
                            f'\tfrom {the_same[1]} parameters \t{100*the_same[0]/the_same[1]:.2f}% were the same')

                res_m.append(res_l)
                hist_m.append(hist_l)
            res_pr.append(res_m)
            hist_pr.append(hist_m)
        res[sparse_from] = np.array(res_pr)
        histograms[sparse_from] = np.array(hist_pr)
    return res, histograms, hist_bins


def check_param_similarity_denses(dense_orig, dense_trained, n_models, verbose=True, which='weight'):
    res, histograms = OrderedDict(), OrderedDict()
    for sparse_from in dense_orig:
        res_pr, hist_pr = [], []
        res_m, hist_m = [], []
        for i in range(n_models[sparse_from]):
            orig_w = get_named_params(
                dense_orig[sparse_from][i], which=which)
            trained_w = get_named_params(
                dense_trained[sparse_from][i], which=which)
            if verbose:
                print(f'{sparse_from}, model {i}')
            res_l, hist_l = [], []
            for j in range(len(orig_w)):
                the_same, hist = check_similarity(
                    orig_w[j], trained_w[j], bins=hist_bins)
                res_l.append(the_same)
                hist_l.append(hist)
                if verbose:
                    print(
                        f'\tfrom {the_same[1]} parameters \t{100*the_same[0]/the_same[1]:.2f}% were the same')
            res_m.append(res_l)
            hist_m.append(hist_l)
            res_pr.append(res_m)
            hist_pr.append(hist_m)
        res[sparse_from] = np.array(res_pr)
        histograms[sparse_from] = np.array(hist_pr)
    return res, histograms, hist_bins


def check_param_similarity(rnd_models, wt_models, n_models, one_shot_pruning_rates, verbose=False, which='weight'):
    res, histograms = OrderedDict(), OrderedDict()
    for sparse_from in rnd_models:
        res_pr, hist_pr = [], []
        for pr in one_shot_pruning_rates:
            res_m, hist_m = [], []
            for i in range(n_models[sparse_from]):
                rnd_w = get_named_params(
                    rnd_models[sparse_from][pr][i], which=which)
                wt_w = get_named_params(
                    wt_models[sparse_from][pr][i], which=which)
                if verbose:
                    print(f'{sparse_from} mask sparsity {pr}, model {i}')
                res_l, hist_l = [], []
                for j in range(len(rnd_w)):
                    the_same, hist = check_similarity(
                        rnd_w[j], wt_w[j], bins=hist_bins)
                    res_l.append(the_same)
                    hist_l.append(hist)
                    if verbose:
                        print(
                            f'\tfrom {the_same[1]} parameters \t{100*the_same[0]/the_same[1]:.2f}% were the same')
                res_m.append(res_l)
                hist_m.append(hist_l)
            res_pr.append(res_m)
            hist_pr.append(hist_m)
        res[sparse_from] = np.array(res_pr)
        histograms[sparse_from] = np.array(hist_pr)
    return res, histograms, hist_bins


# feature distributions

def pruned_feature_distributions(smodels, n_models):
    '''
    How big proportion of the pruned connections are the same?
    '''
    res = OrderedDict()
    for sparse_from in smodels:
        res_m = []
        for ms in list(smodels[sparse_from].keys()):
            res_ms = []
            for i in range(n_models[sparse_from]):
                active_per_input = [buff.numpy().sum(axis=0)
                                    for buff in smodels[sparse_from][ms][i].buffers()]

                res_ms.append(active_per_input)
            res_m.append(res_ms)
        res[sparse_from] = np.array(res_m)

    return res


def input_feature_information(smodel):
    information_masks = []
    for buffer in list(smodel.buffers()):
        mask = buffer.numpy()
        input_sources = mask.sum(axis=1)[:, None]
        input_sources[input_sources == 0] = 1  # don't divide by zero
        information_masks.append(mask / input_sources)

    input_info = information_masks[0]
    for layer_im in information_masks[1:]:
        input_info = layer_im @ input_info

    return input_info.sum(axis=0)


def input_information_distributions(smodels, n_models):
    '''
    How much the last pruned layer has possibility to listen to the input features?
    '''
    res = OrderedDict()
    for sparse_from in smodels:
        res_m = []
        for ms in list(smodels[sparse_from].keys()):
            res_ms = []
            for i in range(n_models[sparse_from]):
                input_info = input_feature_information(
                    smodels[sparse_from][ms][i])
                res_ms.append(input_info)
            res_m.append(res_ms)
        res[sparse_from] = np.array(res_m)

    return res


def pruning_mask_similarity(smodels_a, smodels_b, n_models):
    '''
    How big proportion of the pruned connections are the same?
    '''
    res = OrderedDict()
    for sparse_from in smodels_a:
        res_m = []
        for ms in list(smodels_a[sparse_from].keys()):
            res_ms = []
            for i in range(n_models[sparse_from]):
                buffers_a = list(smodels_a[sparse_from][ms][i].buffers())
                buffers_b = list(smodels_b[sparse_from][ms][i].buffers())

                same_perc = []
                for k, buff_a in enumerate(buffers_a):
                    # how many of the connections were pruned / unpruned the same way?
                    same_mask = buff_a == buffers_b[k]
                    # how many of the pruned connections were the same?
                    same_mask = same_mask[buff_a == 0]
                    proportion = same_mask.sum().item() / same_mask.shape[0]
                    same_perc.append(proportion)
                res_ms.append(same_perc)
            res_m.append(res_ms)
        res[sparse_from] = np.array(res_m)

    return res
