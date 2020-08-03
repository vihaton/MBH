from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .pruning import init_for_pruning, get_named_params, get_pruned_dim, prune_with_predefined_sparsity


def count_models_n_total(n_sizes, smallest_net_densities, n_models):
    return sum([int((n_sizes + 1) * n_sizes / 2 + n_sizes * (len(smallest_net_densities) - 1)) * n_models[key] for key in n_models])


def count_models_n_total_from_hyperparams(hyperparams_for_sparse, n_models):
    n_arc = []
    for name in hyperparams_for_sparse:
        n_m = 0
        for hyperps in hyperparams_for_sparse[name]:
            n_m += len(hyperps)
        n_m *= n_models[name]
        n_arc.append(n_m)

    return sum(n_arc)


def get_sizes(n_sizes, max_hid_neurons, min_hid_neurons, size_scheme='log'):

    if size_scheme == 'linear':
        s_step = (1 - min_hid_neurons / max_hid_neurons)/(n_sizes-1)
        sizes = [1-n*s_step for n in range(n_sizes)]
    elif size_scheme == 'log':
        remain = (min_hid_neurons/max_hid_neurons) ** (1/(n_sizes-1))
        sizes = [remain ** n for n in range(n_sizes)]
    else:
        print('unknown scheme', size_scheme)
        return None

    sizes = [round(s, 3) for s in sizes]
    return sizes


def get_hidden_dims(max_hid_neurons: int):
    assert max_hid_neurons % 4 == 0, 'maximum number of hidden neurons should be compatible with the network architectures'
    last_hidden = max_hid_neurons / 4

    hidden_dims_lenet = (3*last_hidden, last_hidden)
    hidden_dims_dict = {'lenet': hidden_dims_lenet}

    h_dims_pipefc = [last_hidden] * 4
    hidden_dims_dict['pipefc'] = h_dims_pipefc

    hidden_dims_deepfc = (2*last_hidden, last_hidden, last_hidden)
    hidden_dims_dict['deepfc'] = hidden_dims_deepfc

    return hidden_dims_dict


def get_eval_scheme(iterations: int):
    es = [int(1.5**e) for e in range(1+int(np.log(iterations) / np.log(1.5)))]
    es.insert(0, 0)
    if es[-1] != iterations:
        es.append(iterations)

    es = list(set(es))
    es.sort()
    return es


def peek_params(module, verbose=False, print_first=True):
    for name, param_data in module.named_parameters():
        print(name, param_data.shape)
        if verbose and print_first:
            if param_data.dim() > 1:
                print(param_data[0, :10])
            else:
                print(param_data[:10])
        elif verbose:
            print(param_data)


def peek_buffers(module):
    for name, buffer in module.named_buffers():
        print(name, buffer.shape)
        print(buffer)
        print(f'{buffer.sum().item()} / {torch.ones(buffer.shape).sum().item()}')


def print_single_model_info(model):
    params_up = count_unpruned_parameters(model, include_output=True)
    params_up_wo = count_unpruned_parameters(model, include_output=False)
    params_t = count_parameters(model)
    params_wo_out = count_parameters(model, include_output=False)
    neurons = count_neurons(model)
    print(f'\t  hidden neurons\t{neurons}')
    print(f'\t  params per neuron\t{params_up/neurons:.2f}')
    print(
        f'\t  unpruned parameters\t{params_up} ({100*params_up/params_t:.1f}%)')
    print(
        f'\t  unpruned wo output l.\t{params_up_wo} ({100*params_up_wo/params_wo_out:.1f}%)')
    print(
        f'\t  parameters total\t{params_t}\n\t    without output\t{params_wo_out}')
    print()


def print_model_information(models):
    for sparse_from in models:
        print(sparse_from)
        models_p = models[sparse_from]
        for params in models_p:
            print(f'\t#params = {params}')
            for model_h in models_p[params]:
                model = model_h[0]
                print_single_model_info(model)
            print()


def from_output_to_pred(outputs):
    '''outputs are shape [batch_size, classes]'''
    return np.argmax(outputs, axis=-1)


def evaluate(model, data_loader, loss_fn, input_features=28*28, only_first=True):
    '''returns (loss, accuracy)'''
    loss = 0
    correct = 0
    with torch.no_grad():
        for test_d, target in data_loader:
            test_d = test_d.view(-1, input_features)  # flatten
            outputs = model(test_d)
            loss += loss_fn(outputs, target).item()
            preds = from_output_to_pred(outputs)
            correct += (preds == target).sum().item()
            if only_first:
                accuracy = 100 * correct / data_loader.batch_size
                break  # only the first epoch is computed

        if not only_first:
            accuracy = 100 * correct / data_loader.dataset.data.shape[0]
            loss = data_loader.batch_size * loss / \
                data_loader.dataset.data.shape[0]
    return loss, accuracy


def init_models_sparse_vs_dense(hyperparams_for_sparse, n_models, input_features, output_dim, bias_std,
                                random_mask, prune_all_layers, prune_weights):
    models = OrderedDict()

    for sparse_from in hyperparams_for_sparse:
        models_np = OrderedDict()
        for j, hyp_p in enumerate(hyperparams_for_sparse[sparse_from]):
            n_p = hyp_p[0][-1]
            models_d = []
            for dims, d, _ in hyp_p:
                models_n = []
                for i in range(n_models[sparse_from]):
                    if 'lenet' in sparse_from:
                        model = Lenet(
                            input_features, h_dims=dims[:-1], output_dim=output_dim, bias_std=bias_std)
                    elif 'deepfc' in sparse_from:
                        model = DeepFC(
                            input_features, h_dims=dims[:-1], output_dim=output_dim, bias_std=bias_std)
                    elif 'pipefc' in sparse_from:
                        model = DeepFC(
                            input_features, h_dims=dims[:-1], output_dim=output_dim, bias_std=bias_std)
                    else:
                        assert False, 'unknown architecture'

                    model = prune_with_predefined_sparsity(model, density=d,
                                                           random_mask=random_mask,
                                                           prune_all_layers=prune_all_layers,
                                                           unstructured=prune_weights)
                    models_n.append(model)
                models_d.append(models_n)
            models_np[n_p] = models_d
        models[sparse_from] = models_np

    return models


def init_params_normal(tensors: list, std: float):
    for t in tensors:
        nn.init.normal_(t, std=std)


def init_params_constant(tensors: list, c: float):
    for t in tensors:
        nn.init.constant_(t, c)


def init_params_Boris_David(tensors: list, gain=1):
    # https://arxiv.org/pdf/1906.00904.pdf
    # Deep ReLU Networks Have Surprisingly Few Activation Patterns
    for tensor in tensors:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(2.0 / float(fan_in))

        nn.init.normal_(tensor, std=std)


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(nn.Linear, self).__init__(in_features, out_features, bias)
        self.mask = torch.ones(out_features, in_features, requires_grad=False)

    def set_mask(self, mask):
        self.mask = Variable(torch.tensor(mask).float(), requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data

    def get_mask(self):
        return self.mask

    def forward(self, x):
        weight = self.weight*self.mask
        return F.linear(x, weight, self.bias)


class Lenet(nn.Module):
    def __init__(self, input_features, h_dims, output_dim, bias_std=10**-6):
        super(Lenet, self).__init__()

        self.n_hidden_layers = len(h_dims)
        assert self.n_hidden_layers == 2, 'Lenet has 2 hidden dims'
        self.bias_std = bias_std

        self.fc1 = nn.Linear(input_features, h_dims[0])
        self.fc2 = nn.Linear(h_dims[0], h_dims[1])
        self.out = nn.Linear(h_dims[1], output_dim)

        self.init_params()

    def init_params(self):
        # weights
        init_params_Boris_David(get_named_params(
            self, which='weight'
        ),
            gain=1)

        # biases
        init_params_normal(get_named_params(
            self, which='bias'
        ),
            std=self.bias_std)

    def apply_masks(self, masks):
        if len(masks) != self.n_hidden_layers + 1:
            print('there was a wrong number of masks!', len(masks))
            return False

        for i, layer in enumerate(self.children()):
            layer.set_mask(masks[i])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class DeepFC(nn.Module):
    def __init__(self, input_features, h_dims, output_dim, bias_std=10**-6):
        super(DeepFC, self).__init__()
        self.n_hidden_layers = len(h_dims)
        assert self.n_hidden_layers == 3, 'DeepFC has 3 hidden dims'

        self.bias_std = bias_std

        self.fc1 = nn.Linear(input_features, h_dims[0])
        self.fc2 = nn.Linear(h_dims[0], h_dims[1])
        self.fc3 = nn.Linear(h_dims[1], h_dims[2])
        self.out = nn.Linear(h_dims[2], output_dim)

        self.init_params()

    def init_params(self):
        # weights
        init_params_Boris_David(get_named_params(
            self, which='weight'
        ),
            gain=1)

        # biases
        init_params_normal(get_named_params(
            self, which='bias'
        ),
            std=self.bias_std)

    def apply_masks(self, masks):
        if len(masks) != self.n_hidden_layers + 1:
            print('there was a wrong number of masks!', len(masks))
            return False

        for i, layer in enumerate(self.children()):
            layer.set_mask(masks[i])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

# Model saving


def state_dicts_for_dense_models(dense_models):
    state_dicts_dense = OrderedDict()
    for name in dense_models:
        for i, dmodel in enumerate(dense_models[name]):
            state_dicts_dense[f'{name}-{i}_state_dict'] = dmodel.state_dict()
    return state_dicts_dense


def state_dicts_for_sparse_models(models: OrderedDict):
    state_dicts_sparse = OrderedDict()
    for name in models:
        for pr in models[name]:
            for i, smodel in enumerate(models[name][pr]):
                state_dicts_sparse[f'{name}-{pr}-{i}_state_dict'] = smodel.state_dict()
    return state_dicts_sparse

# Model loading


def get_model_list_for_pr(pr, model_dict):
    if pr not in model_dict:
        return []
    else:
        return model_dict[pr]


def dense_models_from_state_dicts(sdicts, input_features, output_dim, hidden_dims_dict, dense_models=None):
    if dense_models is None:
        dense_models = OrderedDict()
    lenets, deepfcs, pipefcs = [], [], []
    for sd_key in sdicts:
        if 'lenet' in sd_key:
            lenet = Lenet(input_features,
                          hidden_dims_dict['lenet'], output_dim)
            lenet.load_state_dict(sdicts[sd_key])
            lenets.append(lenet)
        elif 'deepfc' in sd_key:
            deepfc = DeepFC(
                input_features, hidden_dims_dict['deepfc'], output_dim)
            deepfc.load_state_dict(sdicts[sd_key])
            deepfcs.append(deepfc)
        elif 'pipefc' in sd_key:
            pipefc = DeepFC(
                input_features, hidden_dims_dict['pipefc'], output_dim)
            pipefc.load_state_dict(sdicts[sd_key])
            pipefcs.append(pipefc)

    if len(lenets) > 0:
        dense_models['lenet'] = lenets
    if len(deepfcs) > 0:
        dense_models['deepfc'] = deepfcs
    if len(pipefcs) > 0:
        dense_models['pipefc'] = pipefcs
    return dense_models


def sparse_models_from_state_dicts(sdicts, input_features, output_dim, hidden_dims_dict, prune_weights, sparse_models=None, new_version=False, prune_all_layers=False):
    last_layer_index = len(hidden_dims_dict) + 1 if prune_all_layers else -1

    if sparse_models is None:
        sparse_models = OrderedDict()
    lenets, deepfcs, pipefcs = OrderedDict(), OrderedDict(), OrderedDict()
    for sd_key in sdicts:
        model_name, pr_str, _ = sd_key.split('-')
        pr = float(pr_str)
        hidden_dims = hidden_dims_dict[model_name]
        if not prune_weights and not new_version:  # get pruned dimensions for creating smaller networks
            hidden_dims = [
                get_pruned_dim(pr, h_dim) for h_dim in hidden_dims
            ]

        if model_name == 'lenet':
            smodels_for_pr = get_model_list_for_pr(pr, lenets)
            lenet = Lenet(input_features, hidden_dims, output_dim)
            if new_version:
                init_for_pruning(lenet, ['weight'],
                                 last_layer_index=last_layer_index)
            lenet.load_state_dict(sdicts[sd_key])
            smodels_for_pr.append(lenet)
            lenets[pr] = smodels_for_pr
        elif model_name == 'deepfc':
            smodels_for_pr = get_model_list_for_pr(pr, deepfcs)
            deepfc = DeepFC(input_features, hidden_dims, output_dim)
            if new_version:
                init_for_pruning(deepfc, ['weight'],
                                 last_layer_index=last_layer_index)
            deepfc.load_state_dict(sdicts[sd_key])
            smodels_for_pr.append(deepfc)
            deepfcs[pr] = smodels_for_pr
        elif model_name == 'pipefc':
            smodels_for_pr = get_model_list_for_pr(pr, pipefcs)
            pipefc = DeepFC(input_features, hidden_dims, output_dim)
            if new_version:
                init_for_pruning(pipefc, ['weight'],
                                 last_layer_index=last_layer_index)
            pipefc.load_state_dict(sdicts[sd_key])
            smodels_for_pr.append(pipefc)
            pipefcs[pr] = smodels_for_pr

    if len(list(lenets.keys())) > 0:
        sparse_models['lenet'] = lenets
    if len(list(deepfcs.keys())) > 0:
        sparse_models['deepfc'] = deepfcs
    if len(list(pipefcs.keys())) > 0:
        sparse_models['pipefc'] = pipefcs
    return sparse_models


# Print loaded models


def print_loaded_dense(models):
    print(models.keys())
    for name in models:
        print(f'{name} models {len(models[name])}')


def print_loaded_sparse(models):
    for name in models.keys():
        print(name)
        print(f'\tpruning rates {models[name].keys()}')
        print(f'\tmodels {len(models[name][list(models[name].keys())[0]])}')


# Extract weights

def extract_pruning_mask(layer):
    buffers = list(layer.buffers())
    if len(buffers) == 0:
        pm = None  # this hasn't been pruned yet
    elif len(buffers) == 1:
        pm = buffers[0]
    else:
        raise ValueError('We have more buffers than expected. Expected 1 buffer, got {}'.format(
            len(buffers)))

    return pm


def get_weights(model):
    weights = []
    for child in model.children():
        weights.append(child.weight.detach().numpy())
    return weights


# Accuracies for trained

def compute_acc_for_trained(models, test_loader, loss_fn):
    acc_arc = OrderedDict()
    for name in models:
        acc_p = []
        for params in models[name]:
            acc_h = []
            for models_h in models[name][params]:
                acc_m = np.array([
                    evaluate(model, test_loader, loss_fn)[1]
                    for model in models_h
                ])
                acc_h.append((np.mean(acc_m), np.min(acc_m), np.max(acc_m)))
            acc_p.append(acc_h)
        acc_arc[name] = np.array(acc_p)
    return acc_arc


def compute_acc_for_denses(dense_models_trained, accuracies, test_loader, loss_fn):
    for name in dense_models_trained.keys():
        accs_d = [evaluate(dmodel, test_loader, loss_fn)[1]
                  for dmodel in dense_models_trained[name]]
        accs_d = [np.min(accs_d), np.max(accs_d), np.average(accs_d)]
        accuracies[name] = accs_d
    return accuracies


def compute_acc_for_sparses(sparse_models, one_shot_pruning_rates, test_loader, loss_fn):
    accs_s = OrderedDict()
    for name in sparse_models:
        sms = sparse_models[name]

        accs_pr = []
        for i, pr in enumerate(one_shot_pruning_rates):
            accs_sm = [evaluate(smodel, test_loader, loss_fn)[1]
                       for smodel in sms[pr]]
            accs_pr.append(
                [np.min(accs_sm), np.max(accs_sm), np.average(accs_sm)])

        accs_s[name] = np.array(accs_pr)
    return accs_s


# parameters

def count_parameters(model, include_output=True):
    return sum(p.numel() for name, p in model.named_parameters() if (include_output or 'out' not in name))


def count_unpruned_parameters(model, include_output=True):
    c_param_dict = {
        name.split('_')[0]: param.numel() for name, param in model.named_parameters() if (include_output or 'out' not in name)
    }
    for name, buffer in model.named_buffers():
        if (include_output or 'out' not in name):
            c_param_dict[name.split('_')[0]] = int(buffer.sum().item())

    return sum(c_param_dict.values())


def count_neurons(model):
    # dont count the output layer
    return sum(b.shape[0] for b in get_named_params(model, which='bias')[:-1])


def count_params_from_dims(layer_dims, input_dim, density, count_output=True):
    dims = [input_dim] + layer_dims
    params = 0
    for i in range(1, len(layer_dims)):
        params += int(density * dims[i-1] * dims[i]) + dims[i]

    if count_output:
        params += dims[-2] * dims[-1] + dims[-1]
    return params


def get_architecture_dims(hidden_dims_dict, sizes, output_dim):

    arc_dims = OrderedDict()
    for sparse_from in hidden_dims_dict:
        dims_den = []
        for size in sizes:
            dims = [get_pruned_dim(size, hidden_dims_dict[sparse_from][l])
                    for l in range(len(hidden_dims_dict[sparse_from]))]
            dims += [output_dim]
            dims_den.append(dims)
        arc_dims[sparse_from] = dims_den
    return arc_dims


def get_architecture_n_params(arc_dims, densities, count_output=False):
    arc_params = OrderedDict()
    for sparse_from in arc_dims:
        params_n = []
        for layer_dims in arc_dims[sparse_from]:
            params_den = []
            for density in densities:
                params_den.append(count_params_from_dims(layer_dims, input_dim=input_features,
                                                         density=density,
                                                         count_output=count_output))
            params_n.append(params_den)
        arc_params[sparse_from] = np.array(params_n)
    return arc_params


def get_distance_matrix_to_larges_networks(arc_params):
    dist_to_dense_arc = OrderedDict()

    for sparse_from in arc_params:
        params_len = arc_params[sparse_from]
        distance_to_j = []

        for params_d in params_len[0]:
            distance_to_j.append(np.abs(params_len-params_d))

        dist_to_dense_arc[sparse_from] = np.array(distance_to_j)
    return dist_to_dense_arc


def get_pairs_and_distances(distance_to_large_networks):
    closest_pairs_arc = OrderedDict()
    distances_arc = OrderedDict()

    for sparse_from in distance_to_large_networks:
        distance_to_j = distance_to_large_networks[sparse_from]
        pairs_p = []
        dist_p = []
        for j in range(n_densities):
            pairs_p.append(np.argmin(distance_to_j[j], axis=1))
            dist_min_ji = [distance_to_j[j, i, pairs_p[j][i]]
                           for i in range(1, n_sizes)]
            dist_p.append(dist_min_ji)

        closest_pairs_arc[sparse_from] = np.array(pairs_p)
        distances_arc[sparse_from] = np.array(dist_p)

    return closest_pairs_arc, distances_arc


def get_dense_params(arc_dims, input_features, smallest_net_densities=[1.], count_output=False):
    dense_params = OrderedDict()
    for sparse_from in arc_dims:
        dense_s = []
        for i, dims in enumerate(arc_dims[sparse_from]):
            if i < len(arc_dims[sparse_from]) - 1:
                # not the smallest network
                dense_s.append(count_params_from_dims(dims, input_dim=input_features,
                                                      density=1,
                                                      count_output=count_output))
            else:
                # for the smallest network
                for density in smallest_net_densities:
                    dense_s.append(count_params_from_dims(dims, input_dim=input_features,
                                                          density=density,
                                                          count_output=count_output))

        dense_params[sparse_from] = dense_s
    return dense_params


def get_hyperparams_for_sparses(dense_params, arc_dims, n_models, smallest_net_densities=[1.], count_output=False, input_dim=28**28):
    hyperparams_for_sparse = OrderedDict()

    for sparse_from in dense_params:
        if n_models[sparse_from] < 1:
            continue
        hyp_p = []
        for n_p in dense_params[sparse_from]:
            hyp = []
            for i, dims in enumerate(arc_dims[sparse_from]):
                ds = [1] if i < len(arc_dims[sparse_from]
                                    ) else smallest_net_densities[1:]

                for d in ds:
                    if count_params_from_dims(dims, input_dim, density=d, count_output=count_output) >= n_p:
                        last_l = len(dims) - 1 + int(count_output)
                        input_dims = [input_dim] + dims[:last_l]
                        n_bias = sum(dims[:last_l])
                        n_weights = sum([input_dims[i] * input_dims[i+1]
                                         for i in range(last_l)])
                        d = (n_p - n_bias) / n_weights
                        hyp.append((dims, d, n_p))
            hyp_p.append(hyp)
        hyperparams_for_sparse[sparse_from] = np.array(hyp_p)
    return hyperparams_for_sparse


def print_hyperparams(hyperparams):
    for name in hyperparams:
        for j, hyperps in enumerate(hyperparams[name]):
            print(f'{j+1}: {hyperps[0][-1]/1000:.1f}k parameters')
            for dims, density, params in hyperps:
                print(f'\t layers {dims},\tdensity {density:.2}')


def parse_hidden_neuron_count_from_list_of_hyperparams(hyperp):
    return [sum(h[0][:-1]) for h in hyperp]


def parse_densities_from_list_of_hyperparams(hyperp):
    return [hyperp[i][1] for i in range(len(hyperp))]
