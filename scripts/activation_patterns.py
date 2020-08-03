from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from .models import from_output_to_pred


class hooking_AP():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activation_pattern = (F.relu(output) > 0).numpy().astype(int)

    def close(self):
        self.hook.remove()


class APE():
    '''
    Activation Pattern Extractor
    '''

    def __init__(self, model):
        self.model = model.eval()
        self.layer_patterns = []
        self.activation_patterns = []
        self.predictions = []
        self.targets = []

    def extract_AP(self, inp, target):
        hooks = []
        # dont record activation patterns for the output layer
        for i, kid in enumerate(list(self.model.children())[:-1]):
            hooks.append(hooking_AP(kid))  # one layer pattern hook
        output = self.model.forward(inp)
        self.predictions.append(from_output_to_pred(output).item())
        self.targets.append(target.item())

        # save layer patterns
        lps = []
        layers = list(self.model.children())
        for i, hook in enumerate(hooks):
            lp = np.array(hook.activation_pattern)
            lps.append(lp)
            hook.close()
        self.layer_patterns.append(lps)

        # concatenate activation pattern from layer patterns
        ap = np.concatenate(lps)
        self.activation_patterns.append(ap)

    def get_patterns(self):
        '''returns (APs, LPs, predictions, targets)'''
        lps = np.array(self.layer_patterns)
        aps = np.array(self.activation_patterns)
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        return aps, lps, preds, targets


def activation_pattern_to_str(ap):
    '''numpy array to a compact string'''
    return str(ap)


def aps_to_string(aps):
    aps_str = []
    for ap in aps:
        aps_str.append(activation_pattern_to_str(ap))
    return np.array(aps_str)


def get_APs(model, data_loader, only_first=True, lp_samples=1000, input_features=28*28):
    ape = APE(model)
    c = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(-1, input_features)  # flatten
            for i, image in enumerate(data):
                ape.extract_AP(image, target[i])
                c += 1
                if c == lp_samples:
                    break

            if only_first or c == lp_samples:
                break  # we have reached the limit
    return ape


def get_samples_wrt_postcondition(lps, targets, c: int):
    mask = targets == c
    n_c = mask.sum()
    if n_c == 0:  # there is no sample from this class
        return np.array([[-1]])

    mask = mask if len(mask.shape) == 1 else mask.flatten()

    samples_pc = lps[mask]
    return samples_pc


def samples_under_blanket(samples, constraint_on, constraint_off):
    on_n = constraint_on.sum()
    off_n = constraint_off.sum()
    satisfies_on_constraints = (
        samples[:, constraint_on] == 1).sum(axis=1) == on_n
    satisfies_off_constraints = (
        samples[:, constraint_off] == 0).sum(axis=1) == off_n

    return np.bitwise_and(satisfies_on_constraints, satisfies_off_constraints)


def parse_maximal_pattern_from_constraints(constraint_on, constraint_off):
    neurons_n = len(constraint_on)
    neuron_inds = np.arange(neurons_n)
    neuron_indx = neuron_inds[np.bitwise_xor(constraint_on, constraint_off)]

    activations = (constraint_on[neuron_indx] == 1) * 1
    return np.array((neuron_indx, activations))


def get_neurons_on_xor_off_class_coverage(lps, targets, c, coverage=1.0):
    assert coverage >= 0 and coverage <= 1, 'coverage should be [0,1]'

    # the samples that satisfy the postcondition
    samples_pc = get_samples_wrt_postcondition(lps, targets, c)

    if np.sum(samples_pc) == -1:  # no samples for this class
        return np.array(([-1], [-1]))

    samples_n = samples_pc.shape[0]
    neurons_n = samples_pc.shape[1]

    onoff_ave = np.average(samples_pc == 1, axis=0) - 0.5
    onoff_abs = np.abs(onoff_ave) * 2

    # always on and off
    constraint_on = onoff_ave == 0.5
    constraint_off = onoff_ave == -0.5

    # add constraints while the coverage is over the limit
    sub = samples_under_blanket(samples_pc, constraint_on, constraint_off)
    n_under_blanket = sub.sum()
    current_cov = n_under_blanket / samples_n
    while True:
        #print(f'We can add constraints: enough samples under blanket! Network coverage should be >= {coverage}, the NEURON coverage was {current_cov}')
        smaller_cov_mask = onoff_abs < current_cov
        if smaller_cov_mask.sum() == 0:
            # no smaller coverage, we can break
            break
        # these values have already been used
        onoff_abs[(smaller_cov_mask*1) == 0] = -1
        next_largest_cov_ind = np.argmax(onoff_abs)
        if onoff_abs[next_largest_cov_ind] < coverage:
            # neuron's coverage is smaller than the minimum coverage; we can break
            break

        # add a new constraint, check if the blanket covers enough inputs
        new_const_on = constraint_on.copy()
        new_const_off = constraint_off.copy()

        if onoff_ave[next_largest_cov_ind] > 0:
            # new constraint was on neuron that is on
            new_const_on[next_largest_cov_ind] = 1
        else:
            new_const_off[next_largest_cov_ind] = 1  # neuron was mostly off

        new_sub = samples_under_blanket(
            samples_pc, new_const_on, new_const_off)
        new_cov = new_sub.sum() / samples_n
        if new_cov < coverage:
            # we crossed the line, time to break
            break

        sub = new_sub
        current_cov = onoff_abs[next_largest_cov_ind]
        constraint_on = new_const_on
        constraint_off = new_const_off

    assert sub.sum() / \
        samples_n >= coverage, f'at least {coverage*samples_n:.0f} should have been under the blanket; was {sub.sum()}'

    max_pattern = parse_maximal_pattern_from_constraints(
        constraint_on, constraint_off)

    return max_pattern


def get_max_patterns_for_classes(lpi, targets, n_classes, coverage=1.0):
    mpatterns = []
    for c in range(n_classes):
        mpatterns.append(get_neurons_on_xor_off_class_coverage(
            lpi, targets, c, coverage=coverage))

    return mpatterns


def record_lps_and_max_patterns(model, data_loader, n_samples, n_classes, coverage, input_features=28**2):
    '''return layer patterns, corresponding targets, maximal patterns'''
    # record the APs while feeding test data to the network
    ape = get_APs(model, data_loader, only_first=False,
                  lp_samples=n_samples,
                  input_features=input_features)

    # extract the data from observing the APs earlier
    aps, lps, preds, targets = ape.get_patterns()

    # maximum patterns for specialization data
    # last hidden layer aka -1 (APE doesnt record output layer)
    lpi = parse_layers_patterns_to_numpy(lps, -1)

    # for each class coverage
    max_patterns = []
    for cov in coverage:
        max_patterns.append(get_max_patterns_for_classes(
            lpi, targets, n_classes, coverage=cov))

    return lps, targets, max_patterns


def get_layer_patterns(models, data_loader,
                       coverage,
                       lp_samples, param_keys,
                       n_models, models_total,
                       input_features=28**2, verbose=False):
    prgrss_model = 100 / models_total
    n_classes = len(data_loader.dataset.classes)
    coverage = [coverage] if type(coverage) == int or type(
        coverage) == float else coverage

    n = 0
    lps_arc, targets_arc, maxpat_arc = OrderedDict(), OrderedDict(), OrderedDict()
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

                    lps, targets, max_patterns = record_lps_and_max_patterns(model,
                                                                             data_loader,
                                                                             n_samples=lp_samples,
                                                                             n_classes=n_classes,
                                                                             coverage=coverage,
                                                                             input_features=input_features)

                    lps_ar.append(lps)
                    targets_ar.append(targets)
                    mp_ar.append(max_patterns)

                lps_h.append(lps_ar)
                targets_h.append(targets_ar)
                mp_h.append(mp_ar)

            lps_param.append(lps_h)
            targets_param.append(targets_h)
            mp_param.append(mp_h)

        lps_arc[name] = np.array(lps_param)
        targets_arc[name] = np.array(targets_param)
        maxpat_arc[name] = np.array(mp_param)

    return lps_arc, targets_arc, maxpat_arc


def get_layer_patterns_for_sparse(sparse_models, data_loader, lp_samples, one_shot_pruning_rates, n_models, verbose=False):
    '''
    LT vs WT experiment setup
    '''

    prgrss_architectures = 100 / len(sparse_models.keys())
    prgrss_prs = prgrss_architectures / len(one_shot_pruning_rates)

    lps_arc, targets_arc = OrderedDict(), OrderedDict()
    for i, name in enumerate(sparse_models):
        if verbose:
            print(f'# get patterns for sparse {name}s')
        sms = sparse_models[name]
        prgrss_model = prgrss_prs / n_models[name]

        lps_pr, targets_pr = [], []
        for k, pr in enumerate(one_shot_pruning_rates):
            lps_ar, targets_ar = [], []
            if verbose:
                print(f'## pruning rate {pr}')

            for j, smodel in enumerate(sms[pr]):
                print(
                    f'progress {round(i*prgrss_architectures+k*prgrss_prs+j*prgrss_model,2)}%', end='\r')

                # record the APs while feeding test data to the network
                ape = get_APs(smodel, data_loader, only_first=False,
                              lp_samples=lp_samples,
                              input_features=input_features)

                # extract the data from observing the APs earlier
                aps, lps, preds, targets = ape.get_patterns()

                lps_ar.append(lps)
                targets_ar.append(targets)

            lps_pr.append(lps_ar)
            targets_pr.append(targets_ar)

        lps_arc[name] = np.array(lps_pr)
        targets_arc[name] = np.array(targets_pr)

    return lps_arc, targets_arc


def parse_layers_patterns_to_numpy(lps, layer_i):
    lp_samples = len(lps)
    return np.concatenate(lps[:, layer_i]).reshape((lp_samples, -1))


def calculate_unique_patterns(ape, architecture:str, verbose=True):
    '''
    Now assumes, that every architecture that does not have "lenet" in its name has exactly 3 hidden layers.
    TODO make this more flexible and reliable w.r.t. the number of layers in the architecture. 
    '''
    hidden_layers = ape.model.n_hidden_layers
    aps, lps, preds, targets = ape.get_patterns()

    l1_patterns = np.concatenate(lps[:, 0], axis=0).reshape(len(aps), -1)
    l2_patterns = np.concatenate(lps[:, 1], axis=0).reshape(len(aps), -1)
    if 'lenet' not in architecture:
        l3_patterns = np.concatenate(lps[:, 2], axis=0).reshape(len(aps), -1)

    ul1p = np.unique(l1_patterns, axis=0).shape[0]
    ul2p = np.unique(l2_patterns, axis=0).shape[0]
    if 'lenet' not in architecture:
        ul3p = np.unique(l3_patterns, axis=0).shape[0]

    n_unique_patterns = np.unique(aps, axis=0).shape[0]
    if verbose:
        print(
            f'unique activation patterns #{n_unique_patterns}, {100*n_unique_patterns/len(aps)}')
        print(
            f'unique layer patterns\n\tlayer 1 #{ul1p}\t {100*ul1p/len(aps):.2f}%')
        print(f'\tlayer 2 #{ul2p}\t {100*ul2p/len(aps):.2f}%')
        if 'lenet' not in architecture:
            print(f'\tlayer 3 #{ul3p}\t {100*ul3p/len(aps):.2f}%')
    if 'lenet' not in architecture:
        return n_unique_patterns, [ul1p, ul2p, ul3p]
    else:
        return n_unique_patterns, [ul1p, ul2p]
