import numpy as np
import matplotlib.pyplot as plt
import torch

from .activation_regions import project_to_higher_dim, extract_pruning_mask, extract_bias_and_weights, draw_regions
from .models import parse_hidden_neuron_count_from_list_of_hyperparams, parse_densities_from_list_of_hyperparams

colors = ['k', 'c', 'm', 'r', 'g', 'b', 'y'] * 10
markers = ['o', 's', '^', 'D', 'p', '>', 'P', '<', 'v', '*', 'h'] * 10

# Accuracy


def draw_validation_accuracy_over_specialization(accuracies,
                                                 specialization_data,
                                                 cov_i: int,
                                                 coverages: list,
                                                 sparse_from: str,
                                                 scenario: str,
                                                 hyperparams,
                                                 xscale='linear',
                                                 ylim=(0, 100),
                                                 auto_ylim=False,
                                                 aggregated_spec=False,
                                                 filename=None, save_figures=False):

    alpha = 0.4
    assert cov_i < len(coverages)

    fig = plt.figure(figsize=(8, 6), dpi=180)

    style = ['--', '-', '-.']
    markers = ['o', 's', 'p', '^']

    params = [hyperp[0][-1] for hyperp in hyperparams[sparse_from]]
    neurons = parse_hidden_neuron_count_from_list_of_hyperparams(
        hyperparams[sparse_from][-1])

    leg1, leg2 = [], []
    labels1 = []
    for j, acc_p in enumerate(accuracies[sparse_from]):

        hyperp = hyperparams[sparse_from][j]
        densities = parse_densities_from_list_of_hyperparams(hyperp)

        # prep data ----------------------------------------------------------------------

        # what should we have on the x-axis?
        spec_data_j = specialization_data[scenario][sparse_from][j]
        spec_i = 0
        if aggregated_spec:
            x, x_mi, x_ma = (
                np.mean(spec_data_j[:, :, spec_i, cov_i, i], axis=1) for i in range(3)
            )
        else:
            spec_averaged_over_subspaces = np.mean(
                spec_data_j[:, :, :, cov_i, spec_i], axis=2)
            x = np.mean(spec_averaged_over_subspaces, axis=1)
            x_mi = np.min(spec_averaged_over_subspaces, axis=1)
            x_ma = np.max(spec_averaged_over_subspaces, axis=1)

        acc_d = np.array(accuracies[sparse_from][j])

        # plot -----------------------------------------------------------------------
        if j < len(params):
            if len(acc_d) > 1:
                ave, mi, ma = acc_d[:, 0], acc_d[:, 1], acc_d[:, 2]
            else:
                ave, mi, ma = acc_d[0]

            label = f'#p {params[j]/1000:.1f}k'
            labels1.append(label)

            if len(acc_d) == 1:
                ave, mi, ma = [ave], [mi], [ma]
                x, x_mi, x_ma = [x], [x_mi], [x_ma]

            for k, (s, acc) in enumerate(zip(x, ave)):
                nn = k if k < len(neurons) else len(neurons)-1
                plt.scatter(
                    s, acc, s=10**2, color=f'C{j}', marker=markers[nn], label=label, zorder=3)
                plt.plot([s, s], [mi[k], ma[k]], color=f'C{j}', alpha=alpha)
                plt.plot([x_mi[k], x_ma[k]], [acc, acc],
                         color=f'C{j}', alpha=alpha)

    plt.title(
        f'Validation accuracy over specialization for {sparse_from} models, $c_P = {coverages[cov_i]}$.')
    plt.xscale(xscale)

    if auto_ylim:  # tightens the given ylim if there is slack
        bottom, top = plt.ylim()
        bottom = max(bottom, ylim[0])
        top = min(top, ylim[1])
    else:
        bottom, top = ylim
    plt.ylim((bottom, top))
    plt.xlim(0, 1)

    plt.ylabel('accuracy %')
    plt.xlabel('specialization $s$')

    for n in range(len(params)):
        leg1.append(
            plt.fill(-1, -1, color=f'C{n}')[0]
        )

    first_legend = plt.legend(leg1, labels1)
    plt.gca().add_artist(first_legend)

    for n in range(len(neurons)):
        leg2.append(
            plt.scatter(-1, -1, marker=markers[n],
                        label=f'#h-neurons {neurons[n]}', color='k')
        )

    plt.legend(handles=leg2, loc='lower right')

    plt.grid()
    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_validation_accuracy(accuracies, sparse_from: str, scenario: str,
                             hyperparams,
                             over_neurons, xscale='linear',
                             ylim=(0, 100),
                             auto_ylim=False,
                             filename=None, save_figures=False):

    alpha = 0.1
    assert type(over_neurons) is bool

    fig = plt.figure(figsize=(12, 8))

    style = ['--', '-', '-.']

    if over_neurons:
        # lines share the parameter count
        lines = [hyperp[0][-1]
                 for hyperp in hyperparams[sparse_from]]
    else:  # lines share the neuron count
        lines = parse_hidden_neuron_count_from_list_of_hyperparams(
            hyperparams[sparse_from][-1])

    leg1, leg2 = [], []
    for j, acc_p in enumerate(accuracies[sparse_from]):

        hyperp = hyperparams[sparse_from][j]

        # what should we have on the x-axis?
        if over_neurons:
            x = parse_hidden_neuron_count_from_list_of_hyperparams(hyperp)
            acc_d = np.array(accuracies[sparse_from][j])
        else:
            # draw the lines between models with the same amount of neurons
            # we need to organize the data differently
            x, acc_d = [], []
            # for each number of parameters
            for l, hyperp_p in enumerate(hyperparams[sparse_from]):
                if j < len(hyperp_p):
                    # what was the sparsity of this hyperp setup?
                    x.append(hyperp_p[j][1])
                    # take the corresponding data
                    acc_d.append(accuracies[sparse_from][l][j])
            x, acc_d = np.array(x), np.array(acc_d)

        if j < len(lines):
            if len(acc_d) > 1:
                ave, mi, ma = acc_d[:, 0], acc_d[:, 1], acc_d[:, 2]
            else:
                ave, mi, ma = acc_d[0]

            label = f'#p {lines[j]/1000:.1f}k' if over_neurons else f'#n {lines[j]}'

            if len(acc_d) > 1:
                leg1.append(
                    plt.plot(x, ave, color=colors[j+1], label=label)[0]
                )
                plt.fill_between(x, mi, ma, color=colors[j+1], alpha=alpha)
            else:
                leg1.append(
                    plt.scatter(x, ave, color=colors[j+1], label=label)
                )
                plt.plot([x, x], [mi, ma], color=colors[j+1])

        if not over_neurons:
            # draw lines between the models with same number of params
            params = hyperp[0][-1]
            x_p = parse_densities_from_list_of_hyperparams(hyperp)
            acc_p = np.array(accuracies[sparse_from][j])
            label_params = f'#params {params/1000:.1f}k'

            ave = acc_p[:, 0]

            if len(acc_p) > 1:
                leg2.append(
                    plt.plot(x_p, ave, label=label_params, marker=markers[j], markersize=10,
                             linestyle=style[-1], color='k')[0]
                )
            else:
                leg2.append(
                    plt.scatter(x_p, ave, c='k',
                                label=label_params, marker=markers[j])
                )

    plt.title(
        f'Validation accuracy for {sparse_from} models, {scenario}')
    plt.xscale(xscale)

    if auto_ylim:  # tightens the given ylim if there is slack
        bottom, top = plt.ylim()
        bottom = max(bottom, ylim[0])
        top = min(top, ylim[1])
    else:
        bottom, top = ylim
    plt.ylim((bottom, top))

    plt.ylabel('accuracy %')

    if over_neurons:
        plt.xlabel('#neurons')
        plt.legend()
    else:
        plt.xlabel('density')
        first_legend = plt.legend(handles=leg1, loc='lower left')
        plt.gca().add_artist(first_legend)

        if len(leg2) > 0:
            plt.legend(handles=leg2, loc=(0.1, 0.01))

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_validation_accuracy_from_training(data_from_training, sparse_from: str, evaluation_scheme,
                                           hyperparams_for_sparse,
                                           n_models, xscale='log',
                                           xlim=None,
                                           filename=None, save_figures=False):

    n_params = len(hyperparams_for_sparse[sparse_from])

    for i, params in enumerate(data_from_training[sparse_from]):
        fig = plt.figure(figsize=(12, 7), dpi=160)

        for j, td in enumerate(data_from_training[sparse_from][params]):
            td = td[:, :, 1, 1]  # val acc
            if n_models[sparse_from] > 1:
                ave, mi, ma = np.mean(td, axis=0), np.min(
                    td, axis=0), np.max(td, axis=0)
            else:
                ave, mi, ma = td[0], td[0], td[0]

            dims, d, _ = hyperparams_for_sparse[sparse_from][i][j]

            plt.plot(evaluation_scheme, ave, label=f'{dims[:-1]}-{100*d:.1f}%')
            plt.fill_between(evaluation_scheme, mi, ma, alpha=.2)
        plt.title(
            f'validation accuracy for {sparse_from} with {params/1000:.1f}k parameters')
        plt.xscale(xscale)
        plt.ylim((0, 100))
        plt.ylabel('accuracy %')
        plt.legend()

        xlabel = 'iteration (log)' if 'log' in xscale else 'iteration'
        plt.xlabel(xlabel)
        if xlim is not None:
            plt.xlim(xlim)

        plt.grid()
        plt.tight_layout()
        fig.patch.set_facecolor('w')

        if filename and save_figures:
            plt.savefig(filename.format(round(params/1000)))
        else:
            plt.show()


def draw_plots_from_training(data_from_training, sparse_from: str, one_shot_pruning_rates, evaluate_every_n_iteration, filename=None, save_figures=False):
    '''
    for dense architectures:
        data_from_training[lenet/deepfc] is [model_i, iteration, loss/acc, train/val]

    for sparse:
        data_from_training[sparse] is 
            - ordered dict, key=lenet/deepfc
                - ordered dict, key=pr -> [model_i, iteration, loss/acc, train/val]
    '''
    n_prs = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(14, 4*n_prs+4))
    scores = ['loss', 'accuracy']
    types = ['training', 'validation']
    colors = ['k', 'm', 'y', 'b', 'c']
    c_dict = {
        'lenet': 0,
        'deepfc': 1,
        'sparse rnd': 2,
        'sparse WT': 3,
        'pipefc': 4
    }
    yscales = ['log', 'linear']
    xscales = ['linear', 'linear']

    if isinstance(evaluate_every_n_iteration, list):
        x_ticks = evaluate_every_n_iteration
        xscales = ['linear', 'log']
    else:
        x_ticks = range(0, data_from_training['lenet'].shape[1]
                        * evaluate_every_n_iteration, evaluate_every_n_iteration)

    for i, score in enumerate(scores):
        for j, pr in enumerate(one_shot_pruning_rates):
            ax = plt.subplot(n_prs, 2, j*len(types)+i+1)
            plt.yscale(yscales[i])
            plt.xscale(xscales[i])
            plt.xlabel('iteration')
            plt.ylabel(score)
            plt.title(
                f' validation {score} for {sparse_from}\npruning rate {pr:.3f}, #models {len(data_from_training[sparse_from])}')
            for c, key in enumerate(data_from_training.keys()):
                if key is 'lenet' or key is 'deepfc' or key is 'pipefc':  # the baselines have different data structure
                    if key is not sparse_from:
                        continue  # use only the baseline we have the sparse models from

                    y_mins = np.min(
                        data_from_training[key][:, :, i, 1], axis=0)
                    y_maxs = np.max(
                        data_from_training[key][:, :, i, 1], axis=0)
                    y_ave = np.average(
                        data_from_training[key][:, :, i, 1], axis=0)
                    plt.plot(x_ticks, y_ave,
                             label=f'{key} (unpruned)', alpha=1, c=colors[c_dict[key]])
                    plt.fill_between(x_ticks, y_mins, y_maxs,
                                     alpha=.3, color=colors[c_dict[key]])

                else:  # we have the sparse models
                    y_mins = np.min(
                        data_from_training[key][sparse_from][pr][:, :, i, 1], axis=0)
                    y_maxs = np.max(
                        data_from_training[key][sparse_from][pr][:, :, i, 1], axis=0)
                    y_ave = np.average(
                        data_from_training[key][sparse_from][pr][:, :, i, 1], axis=0)
                    plt.plot(x_ticks, y_ave,
                             label=f'{key}', alpha=1, c=colors[c_dict[key]])
                    plt.fill_between(x_ticks, y_mins, y_maxs,
                                     alpha=.3, color=colors[c_dict[key]])
            ax.legend()

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_accuracies_for_trained(accuracies, sparse_from: str, one_shot_pruning_rates, n_models: {str: int}, blacklist=[], filename=None, xscale='linear', save_figures=False):
    n_prs = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(14, 8))
    plt.xscale(xscale)

    colors = ['k', 'm', 'y', 'b', 'g', 'r', 'c']
    c_dict = {
        'lenet': 0,
        'deepfc': 1,
        'rnd @ init': 2,
        'orig @ init': 3,
        'random trained': 4,
        'original trained': 5,
        'pipefc': 6
    }
    plt.gca().invert_xaxis()

    for c, key in enumerate(accuracies.keys()):
        if key is 'lenet' or key is 'deepfc' or key is 'pipefc':  # the baselines have different data structure
            if key is not sparse_from:
                continue  # use only the baseline we have the sparse models from

            acc_d = np.repeat(accuracies[key], n_prs).reshape((3, -1))

            plt.plot(one_shot_pruning_rates,
                     acc_d[2], label=f'{key} (unpruned)', alpha=1, c=colors[c_dict[key]])
            plt.fill_between(one_shot_pruning_rates,
                             acc_d[0], acc_d[1], alpha=.3, color=colors[c_dict[key]])

        else:  # we have the sparse models
            acc_s = accuracies[key][sparse_from]

            y_min = acc_s[:, 0]
            y_max = acc_s[:, 1]
            y_ave = acc_s[:, 2]
            plt.plot(one_shot_pruning_rates, y_ave,
                     label=f'{key}', alpha=1, c=colors[c_dict[key]])
            plt.fill_between(one_shot_pruning_rates, y_min,
                             y_max, alpha=.3, color=colors[c_dict[key]])

    plt.xlabel('pruning rate')
    plt.ylabel('accuracy')
    plt.title(
        f' validation accuracy for {sparse_from}, #models {n_models[sparse_from]}')
    plt.legend()

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


# Unique activation patterns


def draw_unique_aps_for_sparse(uaps, sparse_from: str, one_shot_pruning_rates, test_batch_size, blacklist=[''], filename=None, xscale='linear', save_figures=False, auto_yscale=False):
    # uaps: key is the scenario, gives OrderedDict key:model -> [#prs, #sparsemodels]
    fig = plt.figure(figsize=(12, 5))
    plt.xscale(xscale)

    colors = ['y', 'b', 'g', 'r']
    plt.gca().invert_xaxis()

    for i, key in enumerate(uaps.keys()):
        if key in blacklist:
            continue

        y_mins = np.min(uaps[key][sparse_from], axis=1)
        y_maxs = np.max(uaps[key][sparse_from], axis=1)
        y_ave = np.average(uaps[key][sparse_from], axis=1)
        plt.plot(one_shot_pruning_rates, y_ave,
                 alpha=1, label=key, c=colors[i])
        plt.fill_between(one_shot_pruning_rates, y_mins,
                         y_maxs, alpha=.3, color=colors[i])

    if not auto_yscale:
        plt.ylim(0, int(1.02*test_batch_size))
    plt.title(f'unique activation patterns for sparse models of {sparse_from}')
    plt.ylabel('unique APs')
    plt.xlabel('pruning rate')
    plt.legend()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_unique_aps(uaps, sparse_from: str, test_batch_size, hyperparams_for_sparse, blacklist=[], filename=None, xscale='linear', save_figures=False, auto_yscale=False):
    '''
        uaps: key is the scenario, gives OrderedDict key:architecture
            -> [n_sizes, #hyperp, #n_models, uaps]

    '''
    fig = plt.figure(figsize=(12, 8))
    plt.xscale(xscale)

    style = ['--', '-']

    for j in range(len(hyperparams_for_sparse[sparse_from])):
        for i, key in enumerate(uaps.keys()):
            if key in blacklist:
                continue

            aps_np = np.array(uaps[key][sparse_from][j])  # aps for param np
            hyperp = np.array(hyperparams_for_sparse[sparse_from][j])

            aps = np.array([(np.mean(m), np.min(m), np.max(m))
                            for m in aps_np])
            neurons_n = parse_hidden_neuron_count_from_list_of_hyperparams(
                hyperp)
            params = hyperp[0][-1]

            ave, mi, ma = aps[:, 0], aps[:, 1], aps[:, 2]

            label = f'{key} #p = {params/1000:.0f}k'
            if len(aps_np) > 1:
                plt.plot(neurons_n, ave, alpha=1,
                         linestyle=style[i], label=label, color=colors[j])
                plt.fill_between(neurons_n, mi, ma, alpha=.05, color=colors[j])
            else:
                plt.scatter(neurons_n, ave, label=label,
                            c=colors[j], marker=markers[i])

    if not auto_yscale:
        plt.ylim(0, int(1.02*test_batch_size))
    plt.title(
        f'unique activation patterns out of {test_batch_size} samples for {sparse_from}')
    plt.ylabel('unique APs')
    plt.legend()

    plt.xlabel('#neurons')

    plt.tight_layout()
    fig.patch.set_facecolor('w')
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


# Unique layer patterns

def draw_unique_lps_sparse(ulps, sparse_from: str, one_shot_pruning_rates, test_batch_size, blacklist=[''], filename=None, xscale='linear', save_figures=False, auto_yscale=False):
    # ulps: key is the scenario, gives OrderedDict key:model -> [#prs, #sparsemodels, #layers]
    if sparse_from is 'lenet':
        n_layers = 2
    else:
        n_layers = 3
    fig = plt.figure(figsize=(12, 4*n_layers))

    colors = ['y', 'b', 'g', 'r']

    for layer in range(n_layers):
        ax = plt.subplot(n_layers, 1, layer+1)
        plt.xscale(xscale)
        plt.gca().invert_xaxis()

        for i, key in enumerate(ulps.keys()):
            if key in blacklist:
                continue
            y_mins = np.min(ulps[key][sparse_from][:, :, layer], axis=1)
            y_maxs = np.max(ulps[key][sparse_from][:, :, layer], axis=1)
            y_ave = np.average(ulps[key][sparse_from][:, :, layer], axis=1)
            plt.plot(one_shot_pruning_rates, y_ave,
                     alpha=1, label=key, c=colors[i])
            plt.fill_between(one_shot_pruning_rates, y_mins,
                             y_maxs, alpha=.3, color=colors[i])

        if not auto_yscale:
            plt.ylim(0, int(1.02*test_batch_size))

        ax.legend()
        plt.title(f'unique layer patterns for {sparse_from}, layer {layer+1}')
        plt.ylabel('unique layer patterns')
        plt.xlabel('pruning rate')

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def get_number_of_neurons_and_the_param_class(hyperparams_for_sparse, sparse_from, param_class_j):
    hyperp = np.array(hyperparams_for_sparse[sparse_from][param_class_j])
    neurons_n = parse_hidden_neuron_count_from_list_of_hyperparams(hyperp)
    params = hyperp[0][-1]
    return (neurons_n, params)


def draw_unique_lps(ulps, sparse_from, test_batch_size, hyperparams_for_sparse,
                    blacklist=[], filename=None, xscale='linear',
                    save_figures=False, auto_yscale=False):
    '''
        # ulps: key is the scenario, gives OrderedDict key:model
            -> [n_sizes, #hyperp, #n_models, #layers, uaps]

    '''
    if sparse_from is 'lenet':
        n_layers = 2
    else:
        n_layers = 3

    style = ['--', '-']
    fig = plt.figure(figsize=(12, 6*n_layers))

    for layer in range(n_layers):
        ax = plt.subplot(n_layers, 1, layer+1)
        plt.xscale(xscale)

        for i, key in enumerate(ulps.keys()):  # before and after training
            if key in blacklist:
                continue

            # for each #p
            for j in range(len(hyperparams_for_sparse[sparse_from])):

                # number of neurons and other hyperparams
                neurons_n, params = get_number_of_neurons_and_the_param_class(hyperparams_for_sparse,
                                                                              sparse_from,
                                                                              param_class_j=j)

                # print('layer', layer, '#parameter', params, 'init/trained', i)
                lps_np = np.array(ulps[key][sparse_from]
                                  [j])  # aps for param np

                lps = np.array([(np.mean(m[:, layer]), np.min(
                    m[:, layer]), np.max(m[:, layer])) for m in lps_np])

                ave, mi, ma = lps[:, 0], lps[:, 1], lps[:, 2]

                label = f'{key} #p = {params/1000:.0f}k'
                if len(lps_np) > 1:
                    plt.plot(neurons_n, ave, alpha=1,
                             linestyle=style[i], label=label, color=colors[j])
                    plt.fill_between(neurons_n, mi, ma,
                                     alpha=.05, color=colors[j])
                else:
                    plt.scatter(neurons_n, ave, label=label,
                                c=colors[j], marker=markers[i])

            if not auto_yscale:
                plt.ylim(0, int(1.02*test_batch_size))

            ax.legend()
            plt.title(
                f'unique layer patterns for {sparse_from}, layer {layer+1}, test samples {test_batch_size}')
            plt.ylabel('unique layer patterns')
            plt.xlabel('#neurons')

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


# Entropy


def draw_entropy(entropy_stats, sparse_from, test_batch_size, hyperparams_for_sparse,
                 blacklist=[], filename=None, xscale='linear',
                 save_figures=False):
    '''
    # entropy_stats: key is the scenario, gives OrderedDict key:model 
        -> [#params, #hyperparams, #models, #stats]
    '''
    fig = plt.figure(figsize=(14, 8))

    distributions = ['predictions', 'true labels']
    stats = ['entropy', 'weighted entropy']
    style = ['--', '-']

    for m, metric in enumerate(stats):
        for k, dist in enumerate(distributions):
            ax = plt.subplot(len(distributions), 2, m*len(distributions)+k+1)
            plt.xscale(xscale)

            for i, key in enumerate(entropy_stats.keys()):
                if key in blacklist:
                    continue
                for j in range(len(hyperparams_for_sparse[sparse_from])):
                    neurons_n, params = get_number_of_neurons_and_the_param_class(hyperparams_for_sparse,
                                                                                  sparse_from,
                                                                                  param_class_j=j)

                    entropy_np = np.array(entropy_stats[key][sparse_from][j])

                    entropy_np_km = entropy_np[:, :, k, m]

                    mi = np.min(entropy_np_km, axis=1)
                    ma = np.max(entropy_np_km, axis=1)
                    ave = np.mean(entropy_np_km, axis=1)

                    label = f'{key} #p = {params/1000:.0f}k'
                    if len(entropy_np_km) > 1:
                        plt.plot(neurons_n, ave, alpha=1,
                                 linestyle=style[i], label=label, color=colors[j])
                        plt.fill_between(neurons_n, mi, ma,
                                         alpha=.05, color=colors[j])
                    else:
                        plt.scatter(neurons_n, ave, label=label,
                                    c=colors[j], marker=markers[i])

            plt.title(f'{metric} of {dist} of sparse {sparse_from}s')
            plt.ylabel(metric)
            plt.xlabel('#neurons')
            ax.legend()

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_entropy_for_sparse(entropy_stats, sparse_from: str, one_shot_pruning_rates, blacklist=[''], filename=None, xscale='linear', save_figures=False):
    '''
    sparse from is lenet or deepfc
    '''
    # entropy_stats: key is the scenario, gives OrderedDict key:model -> [#prs, #sparsemodels, #distributions, #stats]
    fig = plt.figure(figsize=(14, 8))

    distributions = ['predictions', 'true labels']
    stats = ['entropy', 'weighted entropy']
    colors = ['y', 'b', 'g', 'r']

    for j, metric in enumerate(stats):
        for k, dist in enumerate(distributions):
            ax = plt.subplot(len(distributions), 2, j*len(distributions)+k+1)
            plt.xscale(xscale)
            plt.gca().invert_xaxis()

            for i, key in enumerate(entropy_stats.keys()):
                if key in blacklist:
                    continue

                y_mins = np.min(
                    entropy_stats[key][sparse_from][:, :, k, j], axis=1)
                y_maxs = np.max(
                    entropy_stats[key][sparse_from][:, :, k, j], axis=1)
                y_ave = np.average(
                    entropy_stats[key][sparse_from][:, :, k, j], axis=1)

                plt.plot(one_shot_pruning_rates, y_ave,
                         alpha=1, label=key, c=colors[i])
                plt.fill_between(one_shot_pruning_rates, y_mins,
                                 y_maxs, alpha=.3, color=colors[i])

            plt.title(f'{metric} of {dist} of sparse {sparse_from}s')
            plt.ylabel(metric)
            plt.xlabel('pruning rate')
            ax.legend()

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


# Purity


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = int(height) if height > 1 else round(height, 2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(-9, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    size=8)


def get_relative_purity(purity_stats_for_network, unique_aps_for_network):
    max_aps = unique_aps_for_network[:, :, np.newaxis]
    max_aps = np.repeat(max_aps, purity_stats_for_network.shape[-1], axis=2)
    rel = purity_stats_for_network / max_aps
    return rel


def draw_purity_for_sparse(purity_stats, sparse_from: str, one_shot_pruning_rates, blacklist=[], relative=False, uaps=None, filename=None, yscale='linear', save_figures=False):
    n_prs = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(14, 3*n_prs))
    plt.yscale(yscale)

    if uaps is None and relative:
        relative = False
        print('you need number of unique activation patterns to calculate relative amounts, pls define uaps')

    colors = ['y', 'b', 'g', 'r']
    plt.gca().invert_xaxis()
    x_ticks = np.array(list(range(1, 11)))
    n_bars = len(purity_stats.keys()) - len(blacklist)
    width = 0.8 / n_bars
    offsets = [-0.4 + width / 2 + n * width for n in range(n_bars)]
    ylabel = '% of unique activation patterns' if relative else 'number of true labels'
    purity_str = 'relative purity' if relative else 'purity'

    for k, pr in enumerate(one_shot_pruning_rates):
        ax = plt.subplot(len(one_shot_pruning_rates), 1, k+1)
        for i, key in enumerate(purity_stats.keys()):
            if key in blacklist:
                continue

            if relative:
                purities = get_relative_purity(
                    purity_stats[key][sparse_from].view(),
                    uaps[key][sparse_from].view()
                )
            else:
                purities = purity_stats[key][sparse_from].view()

            purity_k = purities[k, :, :]

            purity_min = np.min(purity_k, axis=0)
            purity_max = np.max(purity_k, axis=0)
            purity_ave = np.average(purity_k, axis=0)

            error_low = (purity_ave - purity_min)
            error_high = (purity_max - purity_ave)
            error = np.vstack((error_low, error_high))

            rects = plt.bar(x_ticks + offsets[i], purity_ave, width=width, yerr=error,
                            capsize=7, color=colors[i], edgecolor='k', label=key)
            autolabel(rects, ax)

        plt.title(
            f'{purity_str} of activation patterns for pruning rate {pr} of {sparse_from}')
        plt.ylabel(ylabel)
        plt.xlabel('number of different classes')
        ax.legend()

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()

# changes in weights


def infer_x_range(weight_diffs, sparse_from, wb, by_layers=False):
    x_min, x_max = np.inf, -np.inf
    if by_layers:
        for l, layer in enumerate(weight_diffs[sparse_from]):
            l_min, l_max = np.min(layer[:, wb]), np.max(layer[:, wb])
            x_min = np.min((l_min, x_min))
            x_max = np.max((l_max, x_max))
    else:
        for key in weight_diffs:
            if sparse_from is not key:
                continue
            x_min = min(x_min, np.min(weight_diffs[key][:, wb]))
            x_max = max(x_max, np.max(weight_diffs[key][:, wb]))
    x_range = (x_min, x_max)
    return x_range


def draw_weight_changes_for_whole_network(weight_diffs, sparse_from: str, weights: bool, one_shot_pruning_rates, n_models, draw_baseline=True, x_range=None, auto_range=True, bins=100, blacklist=[], concatenate=True, histtype='bar', filename=None, save_figures=False):
    n_prs = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(12, 3*n_prs))
    colors = ['k', 'm', 'y', 'b']
    c_dict = {
        'lenet': 0,
        'deepfc': 1,
        'rnd': 2,
        'WT': 3
    }

    alphas = {
        'bar': 1,
        'step': 1,
        'stepfilled': 0.3
    }

    if weights:
        wb = 0  # weights
        weights_str = 'weights'
    else:
        wb = 1  # biases
        weights_str = 'biases'

    if x_range is None and not auto_range:
        x_range = infer_x_range(weight_diffs, sparse_from, wb, False)

    for j, pr in enumerate(one_shot_pruning_rates):
        ax = plt.subplot(n_prs, 1, j+1)
        legend = []

        data_to_draw = []
        colors_to_draw = []
        for i, key in enumerate(weight_diffs.keys()):
            if sparse_from not in key:  # from differenet architecture
                continue
            if key is sparse_from and not draw_baseline:
                continue

            for c_key in c_dict:
                if c_key in key:
                    color = colors[c_dict[c_key]]
            colors_to_draw.append(color)

            if key is sparse_from:  # draw the baseline
                wd = weight_diffs[key]
                legend += [f'{key} (unpruned)']
            else:  # draw the sparse networks
                wd = weight_diffs[key][pr]
                legend += [f'{key.split("-")[-1]}-{100*pr:.1f}%']

            if concatenate:
                wd = np.concatenate(wd, axis=1)
                wd = wd[wb]  # weights or biases?
            elif key is sparse_from:  # draw different models separately
                legend = [
                    'x is 0']+[f'{key}{i+1}' for i in range(len(weight_diffs[sparse_from]))]
                wd = wd[:, wb]
                #x_min = np.min(wd[:,wb])
                #x_max = np.max(wd[:,wb])
                #x_range = (x_min, x_max)

            data_to_draw.append(wd)

        if histtype is 'stepfilled' or histtype is 'step':
            legend.reverse()  # but why plt?

        legend = ['x is 0'] + legend

        if concatenate:
            plt.hist(data_to_draw, bins=bins, alpha=alphas[histtype],
                     range=x_range if not auto_range else None,
                     color=colors_to_draw, histtype=histtype, density=True)
            #plt.hist(wd[:,wb].T, bins=bins, range=x_range, alpha=0.3, histtype='stepfilled')

        plt.axvline(0, color='gray')
        plt.xlabel('weight change (trained - init)')
        plt.ylabel('density')
        plt.title(
            f'distribution of changes in {weights_str} for mask sparsity {pr:.3f} {sparse_from}s, #models {n_models[sparse_from]}')
        plt.legend(legend)

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_weight_changes_by_layers(weight_diffs, sparse_from: str, weights: bool, one_shot_pruning_rates, n_models, draw_baseline=False, x_range=None, auto_range=False, bins=100, blacklist=[], concatenate=True, histtype='bar', filename=None, save_figures=False):
    prs = [0] + one_shot_pruning_rates
    n_prs = len(prs)
    colors = ['k', 'm', 'y', 'b']
    c_dict = {
        'lenet': 0,
        'deepfc': 1,
        'rnd': 2,
        'WT': 3,
    }

    alphas = {
        'bar': 1,
        'step': 1,
        'stepfilled': 0.3
    }

    if weights:
        wb = 0  # weights
        weights_str = 'weights'
    else:
        wb = 1  # biases
        weights_str = 'biases'

    legend = ['x is 0']
    if concatenate:
        legend += [sparse_from]
    else:
        legend += [f'{sparse_from}{i+1}' for i in range(
            len(weight_diffs[sparse_from][0]))]

    n_layers = len(weight_diffs[sparse_from])
    fig = plt.figure(figsize=(5*n_layers, 3*n_prs))

    labels = {
        'lenet': ['fc1', 'fc2', 'out'],
        'deepfc': ['fc1', 'fc2', 'fc3', 'out']
    }

    if x_range is None and not auto_range:
        x_range = infer_x_range(weight_diffs, sparse_from, wb, by_layers=True)

    for i, pr in enumerate(prs):
        legend = []
        data_to_draw = [[] for i in range(n_layers)]
        colors_to_draw = []
        for s, key in enumerate(weight_diffs.keys()):
            if sparse_from not in key:  # wrong architecture
                continue
            elif pr > 0 and key is sparse_from and not draw_baseline:  # draw baseline only on the first row
                continue
            elif pr is 0 and key is not sparse_from:  # no sparse data available
                continue

            for c_key in c_dict:
                if c_key in key:
                    color = colors[c_dict[c_key]]
            colors_to_draw.append(color)

            if pr is 0 or (key is sparse_from and draw_baseline):
                wds = weight_diffs[sparse_from]  # unpruned network
                legend += [f'{key} (unpruned)']
            elif key is not sparse_from:
                wds = weight_diffs[key][pr]
                legend += [f'{key.split("-")[-1]}-{100*pr:.1f}%']

            for l, layer in enumerate(wds):
                if concatenate:
                    data_to_draw[l].append(np.concatenate(layer, axis=1)[wb])
                else:
                    data_to_draw[l].append(wds[:, wb])

        if histtype is 'stepfilled' or histtype is 'step':
            legend.reverse()  # but why plt?

        legend = ['x is 0'] + legend

        for l, layers in enumerate(data_to_draw):
            ax = plt.subplot(n_prs, n_layers, i*n_layers+l+1)
            n_neurons = int(len(layers[-1])/n_models[sparse_from])
            sub_bins = int(max(10, n_neurons/30*bins))
            sub_bins = min(bins, sub_bins)

            if concatenate:
                plt.hist(layers,
                         bins=sub_bins,
                         alpha=alphas[histtype],
                         range=x_range,
                         color=colors_to_draw,
                         histtype=histtype,
                         density=True)
            else:
                plt.hist(layer[:, wb].T, bins=bins, alpha=0.25,
                         range=x_range, histtype='stepfilled')
            plt.axvline(0, color='gray')

            if l is 0:
                plt.xlabel('weight change (trained - init)')
                plt.ylabel('density')
            plt.title(
                f'{labels[sparse_from][l]} of {sparse_from}, pr {pr}, #neurons {n_neurons}, #bins {sub_bins}')
        ax.legend(legend)

    plt.suptitle(f'distribution of changes in {weights_str} for {sparse_from}, #models {n_models[sparse_from]}',
                 y=1.01)

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    plt.show()

# parameters


def draw_parameter_distributions(models: list, scenario: str, mask_sparsity: float, n_models=3):
    n_m = len(models)
    n_layers = models[0].n_hidden_layers + 1
    fig = plt.figure(figsize=(4*n_layers+1, 2*n_m + 1))
    colors = ['k', 'm', 'y', 'b', 'c']
    c_dict = {
        'lenet': 0,
        'deepfc': 1,
        'rnd': 2,
        'WT': 3,
        'pipefc': 4
    }
    plt.suptitle(f'{scenario} (mask sparsity {100*mask_sparsity:.1f}%) parameter distributions',
                 y=1.01)

    for i, model in enumerate(models):
        if i == n_models:
            break
        for p, (pname, param) in enumerate(model.named_parameters()):
            ax = plt.subplot(n_m, 2*n_layers, i*2*n_layers+p+1)
            param = param.detach().numpy().flatten()
            bins = min(int(param.size/30), 100)
            bins = max(bins, 10)

            plt.hist(param,
                     bins=bins,
                     color=colors[c_dict[scenario]],
                     density=True)

            plt.title(pname)
            if p == 0:
                plt.ylabel(f'net {i+1} parameter densities')

            plt.axvline(0, color='gray')
    plt.tight_layout()
    fig.patch.set_facecolor('w')


def draw_sparse_parameter_distributions(random_sm, wt_sm, one_shot_pruning_rates, sparse_from='lenet', n_models=3):
    for pr in one_shot_pruning_rates:
        draw_parameter_distributions(
            random_sm[sparse_from][pr], 'rnd', pr, n_models)
        draw_parameter_distributions(
            wt_sm[sparse_from][pr], 'WT', pr, n_models)


def draw_parameter_similarity_results(res, yscale='linear', title='', filename=None, save_figures=False):
    '''
    res is OrderedDict key=architecture
            [#pr, #models, #layers, (#different params, #total params)]
    '''
    fig = plt.figure(figsize=(14, 5))

    for key in res:
        for l in range(res[key].shape[-2]):
            same = res[key][:, :, l, 0].flatten()
            total = res[key][:, :, l, 1].flatten()
            perc = 100*same/total
            plt.plot(perc, label=f'{key} l{l+1}')

    plt.title(f'{title}\npercentage of weights that are exactly the same floats')
    plt.ylim(-1, 103)
    plt.xlabel('models from small to big pruning')
    plt.legend()
    plt.yscale(yscale)

    fig.patch.set_facecolor('w')
    plt.tight_layout()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_parameter_similarity_histograms(histograms, bin_edges, density=False, yscale='linear', title='', filename=None, save_figures=False):
    '''
    histograms is an OrderedDict key=architecture
            [#pr, #models, #layers, histograms of the same values]
    '''
    n_arch = len(list(histograms.keys()))
    fig = plt.figure(figsize=(14, 4*n_arch))

    for j, key in enumerate(histograms.keys()):
        plt.subplot(n_arch, 1, j + 1)
        n_layers = histograms[key].shape[2]
        # assumes bins are equal width!
        bin_width = (abs(bin_edges[1] - bin_edges[0]) / n_layers)
        bin_width *= 0.9
        bin_locations = [np.array(
            bin_edges[:-1]) + n * bin_width + bin_width / 2 for n in range(n_layers)]

        for l in range(n_layers):
            hist_same_values = histograms[key][:, :, l].sum(
                axis=0).sum(axis=0)  # sum over pr:s, then over models
            if density:
                n_params_total = hist_same_values.sum()
                hist_same_values = hist_same_values / n_params_total * 100
            plt.bar(x=bin_locations[l], height=hist_same_values,
                    width=bin_width, label=f'{key} l{l+1}')

        plt.title(f'{title}\nhistogram of the values that were the same')
        plt.xlabel('parameter values, that were exactly the same')
        ylabel = 'number of parameters' if not density else 'proportion of parameters (%)'
        plt.ylabel(ylabel)
        plt.legend()
        plt.yscale(yscale)

    fig.patch.set_facecolor('w')
    plt.tight_layout()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


# activation regions

def draw_activation_regions_1D(region_count_data, sparse_from: str, one_shot_pruning_rates: [], draw_dense: bool, by_layer: bool, average_over_images: int, n_models, blacklist=[], wide_interval=True, filename=None, xscale='linear', save_figures=False):
    '''
    region_count_data: key is the scenario, gives OrderedDict key:model -> [#prs, (ave, min, max)]

    '''
    n_prs = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(16, 8))
    plt.xscale(xscale)

    colors = ['k', 'm', 'y', 'g', 'b', 'r']

    plt.gca().invert_xaxis()

    def get_min_max(y):
        if wide_interval:  # take the averages of min and max values that occurred in the `average_over_images` images we calculated the ARs over
            return y[:, 1], y[:, 2]
        else:  # take the min and max of averages
            assert False, 'The data doesnt allow this, needs to be implemented!'
            return np.min(y[:, 0], axis=1), np.max(y[:, 0], axis=1)

    for i, key in enumerate(region_count_data.keys()):
        if key in blacklist:
            continue

        for j, name in enumerate(region_count_data[key].keys()):

            if sparse_from not in name:
                continue

            if 'dense' in key:  # draw baseline
                if not draw_dense:
                    continue
                y = region_count_data[key][name].repeat(n_prs).reshape(3, -1)
                mi, ma = np.min(y), np.max(y)

                train_str = key.split('-')[-1]

                plt.plot(one_shot_pruning_rates,
                         y[0], label=f'{name}-{train_str} (unpruned)', alpha=1)  # , c=colors_d[j])
                plt.fill_between(one_shot_pruning_rates,
                                 mi, ma, alpha=.2)  # , color=colors_d[j])

            else:  # draw sparse models
                y = region_count_data[key][name]
                mi, ma = get_min_max(y)

                plt.plot(one_shot_pruning_rates,
                         y[:, 0],
                         alpha=1, label=key, c=colors[i])
                plt.fill_between(one_shot_pruning_rates,
                                 mi,
                                 ma, alpha=.2, color=colors[i])

    plt.title(
        f'Number of local, 1D activation regions by {sparse_from}s\naveraged over {average_over_images} images and {n_models[sparse_from]} models')
    plt.ylabel('#ARs')
    plt.xlabel('mask sparsity')
    plt.legend()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_ARs_from_training(ar_data_from_training, sparse_from: str, one_shot_pruning_rates, evaluation_scheme, hidden_dims_dict, average_over_images: int, n_models: dict, xscale='log', wide_interval=False, filename=None, save_figures=False):
    '''
    Plots the number of activation regions through training for the whole network (not by layer).

    for dense architectures:
        [sparse from](regions, ars, arls)
            [#models, #evaluations, (ave, min, max)]

    for sparse:
        ar_data_from_training[sparse] is 
            [sparse from][sparsity](regions, ars, arls)
                [#models, #evaluations, (ave, min, max)]    
    '''
    n_prs = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(14, 4*n_prs+4))
    colors = ['k', 'm', 'y', 'b', 'c']
    c_dict = {
        'lenet': 0,
        'deepfc': 1,
        'sparse rnd': 2,
        'sparse WT': 3,
        'pipefc': 4
    }

    x_ticks = evaluation_scheme
    interval_str = 'wide interval (averages of min and max)' if wide_interval else 'narrow interval (min and max of averages)'

    def get_min_max(y):
        if wide_interval:  # take the averages of min and max values that occurred in the `average_over_images` images we calculated the ARs over
            return np.average(y[:, :, 1], axis=0), np.average(y[:, :, 2], axis=0)
        else:  # take the min and max of averages
            return np.min(y[:, :, 0], axis=0), np.max(y[:, :, 0], axis=0)

    for j, pr in enumerate(one_shot_pruning_rates):
        ax = plt.subplot(n_prs, 1, j+1)
        plt.yscale('linear')
        plt.xscale(xscale)
        plt.xlabel('iteration')
        plt.ylabel('#ARs')
        plt.title(
            f'number of 1D activation regions for {sparse_from} averaged over {average_over_images} images\nmask sparsity {pr:.2f}, #models {n_models[sparse_from]}, {interval_str}')
        for c, key in enumerate(ar_data_from_training.keys()):
            if key is 'lenet' or key is 'deepfc' or key is 'pipefc':  # the baselines have different data structure
                if key is not sparse_from:
                    continue  # use only the baseline we have the sparse models from
                ars = ar_data_from_training[key][1]

                y_mins, y_maxs = get_min_max(ars)
                y_ave = np.average(
                    ars[:, :, 0], axis=0)
                plt.plot(x_ticks, y_ave,
                         label=f'{key} unpruned {hidden_dims_dict[key]}', alpha=1, c=colors[c_dict[key]])
                plt.fill_between(x_ticks, y_mins, y_maxs,
                                 alpha=.3, color=colors[c_dict[key]])

            else:  # we have the sparse models
                ars = ar_data_from_training[key][sparse_from][pr][1]
                y_mins, y_maxs = get_min_max(ars)
                y_ave = np.average(
                    ars[:, :, 0], axis=0)
                plt.plot(x_ticks, y_ave,
                         label=f'{key}', alpha=1, c=colors[c_dict[key]])
                plt.fill_between(x_ticks, y_mins, y_maxs,
                                 alpha=.2, color=colors[c_dict[key]])
        ax.legend()
        plt.ylim(bottom=0)

    plt.tight_layout()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()

    '''
    Plots the number of activation regions through training for the whole network (not by layer).

    for dense architectures:
        [sparse from](regions, ars, arls)
            [#models, #evaluations, (ave, min, max)]

    for sparse:
        ar_data_from_training[sparse] is 
            [sparse from][sparsity](regions, ars, arls)
                [#models, #evaluations, (ave, min, max)]    
    '''
    n_prs = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(14, 4*n_prs+4))
    colors = ['k', 'm', 'y', 'b', 'c']
    c_dict = {
        'lenet': 0,
        'deepfc': 1,
        'sparse rnd': 2,
        'sparse WT': 3,
        'pipefc': 4
    }

    x_ticks = evaluation_scheme

    def get_min_max(y):
        if wide_interval:  # take the averages of min and max values that occurred in the `average_over_images` images we calculated the ARs over
            return np.average(y[:, :, 1], axis=0), np.average(y[:, :, 2], axis=0)
        else:  # take the min and max of averages
            return np.min(y[:, :, 0], axis=0), np.max(y[:, :, 0], axis=0)

    for j, pr in enumerate(one_shot_pruning_rates):
        ax = plt.subplot(n_prs, 1, j+1)
        plt.yscale('linear')
        plt.xscale(xscale)
        plt.xlabel('iteration')
        plt.ylabel('#ARs')
        plt.title(
            f'number of 1D activation regions for {sparse_from} averaged over {average_over_images} images\nmask sparsity {pr:.2f}, #models {n_models[sparse_from]}')
        for c, key in enumerate(ar_data_from_training.keys()):
            if key is 'lenet' or key is 'deepfc' or key is 'pipefc':  # the baselines have different data structure
                if key is not sparse_from:
                    continue  # use only the baseline we have the sparse models from
                ars = ar_data_from_training[key][1]

                y_mins = np.min(
                    ars[:, :, 0], axis=0)
                y_maxs = np.max(
                    ars[:, :, 0], axis=0)
                y_ave = np.average(
                    ars[:, :, 0], axis=0)
                plt.plot(x_ticks, y_ave,
                         label=f'{key} unpruned {hidden_dims_dict[key]}', alpha=1, c=colors[c_dict[key]])
                plt.fill_between(x_ticks, y_mins, y_maxs,
                                 alpha=.3, color=colors[c_dict[key]])

            else:  # we have the sparse models
                ars = ar_data_from_training[key][sparse_from][pr][1]
                y_mins = np.min(
                    ars[:, :, 0], axis=0)
                y_maxs = np.max(
                    ars[:, :, 0], axis=0)
                y_ave = np.average(
                    ars[:, :, 0], axis=0)
                plt.plot(x_ticks, y_ave,
                         label=f'{key}', alpha=1, c=colors[c_dict[key]])
                plt.fill_between(x_ticks, y_mins, y_maxs,
                                 alpha=.2, color=colors[c_dict[key]])
        ax.legend()
        plt.ylim(bottom=0)

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_ARs_by_layer_from_training(ar_data_from_training, sparse_from: str, one_shot_pruning_rates, evaluation_scheme, hidden_dims_dict, average_over_images: int, n_models: dict, xscale='log', filename=None, save_figures=False):
    '''
    Plots the number of regions by layers.

    for dense models
        [key](regions, ars, arls)
            [#models, #evaluations, #layers, (ave, min, max)]

    for sparse models
        [key][sparse from][mask sparsity](regions, ars, arls)
            [#models, #evaluations, #layers, (ave, min, max)]
    '''
    n_prs = len(one_shot_pruning_rates)
    n_layers = ar_data_from_training[sparse_from][2].shape[-2]
    colors = ['k', 'm', 'y', 'b', 'c']
    c_dict = {
        'lenet': 0,
        'deepfc': 1,
        'sparse rnd': 2,
        'sparse WT': 3,
        'pipefc': 4
    }

    linestyles = ['-', '--', '-.', ':']

    x_ticks = evaluation_scheme

    for j, pr in enumerate(one_shot_pruning_rates):
        fig = plt.figure(figsize=(14, 3*n_layers+5))
        plt.suptitle(f'Number of 1D activation regions for {sparse_from} averaged over {average_over_images} images, #models {n_models[sparse_from]}',
                     y=1.01)

        for layer in range(n_layers):
            ax = plt.subplot(n_layers, 1, layer+1)
            plt.yscale('linear')
            plt.xscale(xscale)
            plt.xlabel('iteration')
            plt.ylabel('#ARs')
            plt.title(
                f'mask sparsity {pr:.2f}, layer {layer+1}'
            )

            for c, key in enumerate(ar_data_from_training.keys()):
                if key is 'lenet' or key is 'deepfc' or key is 'pipefc':  # the baselines have different data structure
                    if key is not sparse_from:
                        continue  # use only the baseline we have the sparse models from

                    arls = ar_data_from_training[key][2]
                    y_mins = np.min(
                        arls[:, :, layer, 0], axis=0)
                    y_maxs = np.max(
                        arls[:, :, layer, 0], axis=0)
                    y_ave = np.average(
                        arls[:, :, layer, 0], axis=0)
                    plt.plot(x_ticks, y_ave,
                             linestyle=linestyles[layer],
                             label=f'{key} unpruned {hidden_dims_dict[key]}',
                             alpha=1, c=colors[c_dict[key]])
                    plt.fill_between(x_ticks, y_mins, y_maxs,
                                     linestyle=linestyles[layer],
                                     alpha=.3, color=colors[c_dict[key]])

                else:  # we have the sparse models
                    arls = ar_data_from_training[key][sparse_from][pr][2]
                    y_mins = np.min(
                        arls[:, :, layer, 0], axis=0)
                    y_maxs = np.max(
                        arls[:, :, layer, 0], axis=0)
                    y_ave = np.average(
                        arls[:, :, layer, 0], axis=0)
                    plt.plot(x_ticks, y_ave,
                             ls=linestyles[layer],
                             label=f'{key}', alpha=1, c=colors[c_dict[key]])
                    plt.fill_between(x_ticks, y_mins, y_maxs,
                                     ls=linestyles[layer],
                                     alpha=.2, color=colors[c_dict[key]])
            ax.legend()
            plt.ylim(bottom=0)

        fig.patch.set_facecolor('w')
        plt.tight_layout()
        if filename and save_figures:
            plt.savefig(filename.format(j))
        else:
            plt.show()


def concatenate_over_images(ar_distribution, model_i, iteration, layer):
    # models, evaluations, images, layers, AR locations
    return np.concatenate(ar_distribution[model_i, iteration, :, layer])


def draw_1D_AR_border_distributions(ar_data_from_training, sparse_from: str, one_shot_pruning_rates, evaluation_scheme, hidden_dims_dict, average_over_images, yscale='linear', density=False, filename=None, save_figures=False):
    '''
    Plots the distribution of 1D region borders by layers.

    for dense models
        [key](regions, ars, arls)
            [models, evaluations, images, layers]
                        [AR locations]
    for sparse models
        [key][sparse from][mask sparsity](regions, ars, arls)
            [models, evaluations, images, layers]
                        [AR locations]
    '''
    n_prs = len(one_shot_pruning_rates)
    n_layers = ar_data_from_training[sparse_from][0].shape[-1]

    n_evaluations = len(evaluation_scheme)
    n_models = min(ar_data_from_training[sparse_from][0].shape[0], 4)

    colors = ['k', 'm', 'y', 'b', 'c']
    c_dict = {
        'lenet': 0,
        'deepfc': 1,
        'sparse rnd': 2,
        'sparse WT': 3,
        'pipefc': 4
    }

    linestyles = ['-', '--', '-.', ':']

    for j, ms in enumerate(one_shot_pruning_rates):

        fig = plt.figure(figsize=(18, 2+5*n_evaluations))
        plt.suptitle(
            f'AR border distribution during training, mask sparsity {round(100*ms)}', y=1.00)

        for i, iteration in enumerate(evaluation_scheme):
            for m in range(n_models):
                if m == 3:
                    break
                cols = min(3, n_models)
                ax = plt.subplot(n_evaluations, cols, i*cols+m+1)
                plt.title(
                    f'Distribution of 1D AR borders\niteration {iteration}, model {m}, mask sparsity {100*ms:.1f}')
                plt.yscale(yscale)

                for c, key in enumerate(ar_data_from_training.keys()):
                    if key is 'lenet' or key is 'deepfc' or key is 'pipefc':  # the baselines have different data structure
                        if key is not sparse_from:
                            continue  # use only the baseline we have the sparse models from
                        regions = ar_data_from_training[key][0]

                        # this loses infro about layers!
                        aggregate = np.concatenate(
                            np.concatenate(regions[m, i, :, :]))
                        plt.hist(aggregate, bins=100, range=(0, 1), density=density, alpha=0.5,
                                 label=f'{key} unpruned {hidden_dims_dict[key]}', color=colors[c_dict[key]])

                    else:  # we have the sparse models
                        regions = ar_data_from_training[key][sparse_from][ms][0]
                        aggregate = np.concatenate(
                            np.concatenate(regions[m, i, :, :]))
                        plt.hist(aggregate, bins=100, range=(0, 1), density=density, alpha=0.5,
                                 label=f'{key}', color=colors[c_dict[key]])

                ax.legend()
                # plt.ylim(bottom=0)

        fig.patch.set_facecolor('w')
        plt.tight_layout()
        if filename and save_figures:
            plt.savefig(filename.format(j))
        else:
            plt.show()


def draw_activation_regions_1D_sparse_vs_dense(region_count_data, sparse_from, average_over_images,
                                               by_layer, n_sizes, hyperparams_for_sparse, n_models, xscale='linear',
                                               wide_interval=True, blacklist=[], filename=None, save_figures=False):
    '''
    region_count_data: key is the scenario, gives OrderedDict key:model
        by_layer == False
            -> [#params, #hyperparams, (ave, min, max)]
        by_layer == True
            -> [#params, #hyperparams, #layers, (ave, min, max)]
    '''

    # TODO by layers

    fig = plt.figure(figsize=(16, 8))
    style = ['--', '-']

    def get_min_max(y):
        if wide_interval:  # take the averages of min and max values that occurred in the `average_over_images` images we calculated the ARs over
            return y[:, 1], y[:, 2]
        else:  # take the min and max of averages
            assert False, 'The data doesnt allow this, needs to be implemented!'
            return np.min(y[:, 0], axis=1), np.max(y[:, 0], axis=1)

    for k, key in enumerate(region_count_data.keys()):
        if key in blacklist:
            continue

        for i, name in enumerate(region_count_data[key].keys()):

            if sparse_from not in name:
                continue

            for j in range(n_sizes):
                neurons_n, params = get_number_of_neurons_and_the_param_class(hyperparams_for_sparse,
                                                                              sparse_from,
                                                                              param_class_j=j)

                y = region_count_data[key][name][j]

                y = np.array(y)

                mi, ma = get_min_max(y)

                label = f'{key} #p = {params/1000:.0f}k'
                if len(y) > 1:
                    plt.plot(
                        neurons_n, y[:, 0], alpha=1, linestyle=style[k], label=label, color=colors[j])
                    plt.fill_between(neurons_n, mi, ma,
                                     alpha=.05, color=colors[j])
                else:
                    plt.scatter(
                        neurons_n, y[:, 0], label=label, c=colors[j], marker=markers[k])

    plt.title(
        f'Number of local, 1D activation regions by {sparse_from}s\naveraged over {average_over_images} images and {n_models[sparse_from]} models')
    plt.ylabel('#ARs')
    plt.xlabel('#neurons')
    plt.legend()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


# 2D ARs


def draw_number_of_2D_ARs(ar_data_2D, sparse_from, normalized_by_area: bool,
                          average_over_images, hyperparams_for_sparse, n_models,
                          param_keys, over_neurons=True,
                          wide_interval=False, just_area=False, draw_init=False, use_three_samples=None,
                          filename=None, save_figures=False):
    '''
    ar data 2D is Ordered dict key:scenario
        -> OD key:model
            -> [#n_sizes]
                -> [#hyperp, #models, #subspaces, normalizedAR/AR/area]
    '''

    fig = plt.figure(figsize=(10, 6), dpi=160)

    scenarios = ar_data_2D.keys()
    if not draw_init:
        scenarios = ['trained']  # not flexible

    style = ['--', '-', '-.']

    leg1, leg2 = [], []

    stats_i = 0
    if just_area:
        stats_i = 2
    elif not normalized_by_area:
        stats_i = 1

    if over_neurons:
        lines = param_keys[sparse_from]  # lines share the parameter count
    else:  # lines share the neuron count
        lines = parse_hidden_neuron_count_from_list_of_hyperparams(
            hyperparams_for_sparse[sparse_from][-1])

    def get_min_max(stats, stats_i, wide_interval):
        if wide_interval:  # averages of subspace min and max
            mi, ma = np.min(stats[:, :, :, stats_i], axis=2), np.max(
                stats[:, :, :, stats_i], axis=2)
            return np.mean(mi, axis=1), np.mean(ma, axis=1)
        else:  # min and max of the subspace averages
            ave = np.mean(stats[:, :, :, stats_i], axis=2)
            return np.min(ave, axis=1), np.max(ave, axis=1)
    for i, scenario in enumerate(scenarios):
        subi = 0 if 'init' in scenario else 1

        if use_three_samples is None:
            # we might have both types of 2d planes,
            # through the origin and with three images.
            # Infer that from the scenario name.
            origi = 0 if 'orig' in scenario else 1
        else:
            origi = 0 if not use_three_samples else 1

        for j in range(len(hyperparams_for_sparse[sparse_from])):

            hyperp = hyperparams_for_sparse[sparse_from][j]

            # what should we have on the x-axis?
            if over_neurons:
                x = np.array(parse_hidden_neuron_count_from_list_of_hyperparams(
                    hyperp))
                stats = ar_data_2D[scenario][sparse_from][j]

            else:
                # draw the lines between models with the same amount of neurons
                # we need to organize the data differently
                x, stats = [], []
                # for each number of parameters
                for l, hyperp_p in enumerate(hyperparams_for_sparse[sparse_from]):
                    # if we have this architecture
                    if j < len(hyperp_p):
                        # what was the density of this hyperp setup?
                        x.append(hyperp_p[j][1])
                        # take the corresponding data
                        stats.append(ar_data_2D[scenario][sparse_from][l][j])
                x, stats = np.array(x), np.array(stats)

            params = hyperp[0][-1]
            neurons = sum(hyperparams_for_sparse[sparse_from][j][-1][0][:-1])

            if j < len(lines):
                # take mean of subspaces...
                ave = np.mean(stats[:, :, :, stats_i], axis=2)
                # ...and models
                ave = np.mean(ave, axis=1)
                mi, ma = get_min_max(stats, stats_i, wide_interval)

                scstr = f'{scenario} ' if len(scenarios) > 1 else ''
                pstr = f'#p {params/1000:.1f}k' if over_neurons else f'#n {neurons}'

                label = f'{scstr}{pstr}'
                if len(stats) > 1:
                    leg1.append(
                        plt.plot(x, ave, label=label,
                                 linestyle=style[subi], color=colors[j+1])[0]
                    )
                    plt.fill_between(x, mi, ma, alpha=0.1, color=colors[j+1])
                else:
                    leg1.append(
                        plt.scatter(
                            x, ave, c=colors[j+1], marker=markers[i], label=label)
                    )
                    plt.plot([x, x], [mi, ma], c=colors[j+1])

            if not over_neurons:
                x_params = np.array(
                    parse_densities_from_list_of_hyperparams(hyperp))
                stats_p = ar_data_2D[scenario][sparse_from][j]
                label_params = f'#params {params/1000:.1f}k'

                ave = np.mean(stats_p[:, :, :, stats_i], axis=2)
                ave = np.mean(ave, axis=1)

                if len(stats_p) > 1:
                    leg2.append(
                        plt.plot(x_params, ave, label=label_params, marker=markers[j], markersize=10,
                                 linestyle=style[-1], color='k')[0]
                    )
                else:
                    leg2.append(
                        plt.scatter(x_params, ave, c='k',
                                    label=label_params, marker=markers[i])
                    )

    bot, top = plt.ylim()
    plt.ylim(bottom=-0.05*top)

    if not just_area:

        first_legend = plt.legend(handles=leg1, loc='upper left')
        plt.gca().add_artist(first_legend)

        if len(leg2) > 0:
            plt.legend(handles=leg2, loc='upper right')
            plt.grid()

        ylabel = '#local 2D ARs' if not normalized_by_area else '#local 2D ARs / area of the local 2D plane'

        wstr = 'wide interval (averages of min & max)' if wide_interval else 'narrow interval (min & max of averages)'
        astr = 'normalized by area' if normalized_by_area else 'absolute numbers'
        trained_str = '' if len(scenarios) > 1 else 'after training'
        plt.title(
            f'Number of local 2D activation regions ({astr}) {trained_str} of {sparse_from}\naveraged over {average_over_images} 2D planes and {n_models[sparse_from]} models, {wstr}')
    else:
        ylabel = '2D plane area'
        plt.title(
            'Area of the 2 dimensional plane we counted the local ARs on')
        plt.legend()

    plt.ylabel(ylabel)

    if over_neurons:
        plt.xlabel('#neurons')
    else:
        plt.xlabel('density')

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_number_of_2D_ARs_aggregated(ar_data_2D, sparse_from, normalized_by_area: bool,
                                     average_over_images, hyperparams_for_sparse, n_models,
                                     param_keys, over_neurons=True,
                                     wide_interval=False, just_area=False, draw_init=False, use_three_samples=None,
                                     filename=None, save_figures=False):
    '''
    ar data 2D is Ordered dict key:scenario
        -> OD key:model
            -> [#n_sizes]
                -> [#hyperp, #models, normalizedAR/AR/area, ave/min/max]
    '''

    fig = plt.figure(figsize=(10, 6), dpi=160)

    scenarios = ar_data_2D.keys()
    if not draw_init:
        scenarios = ['trained']  # not flexible

    style = ['--', '-', '-.']

    leg1, leg2 = [], []

    stats_i = 0
    if just_area:
        stats_i = 2
    elif not normalized_by_area:
        stats_i = 1

    if over_neurons:
        lines = param_keys[sparse_from]  # lines share the parameter count
    else:  # lines share the neuron count
        lines = parse_hidden_neuron_count_from_list_of_hyperparams(
            hyperparams_for_sparse[sparse_from][-1])

    def get_min_max(stats, stats_i, wide_interval=False):
        if wide_interval:  # averages of min and max
            return np.mean(stats[:, :, stats_i, 1], axis=1), np.mean(stats[:, :, stats_i, 2], axis=1)
        else:  # min and max of the averages
            return np.min(stats[:, :, stats_i, 0], axis=1), np.max(stats[:, :, stats_i, 0], axis=1)

    for i, scenario in enumerate(scenarios):
        subi = 0 if 'init' in scenario else 1

        if use_three_samples is None:
            # we might have both types of 2d planes,
            # through the origin and with three images.
            # Infer that from the scenario name.
            origi = 0 if 'orig' in scenario else 1
        else:
            origi = 0 if not use_three_samples else 1

        for j in range(len(hyperparams_for_sparse[sparse_from])):

            hyperp = hyperparams_for_sparse[sparse_from][j]

            # what should we have on the x-axis?
            if over_neurons:
                x = np.array(parse_hidden_neuron_count_from_list_of_hyperparams(
                    hyperp))
                stats = ar_data_2D[scenario][sparse_from][j]

            else:
                # draw the lines between models with the same amount of neurons
                # we need to organize the data differently
                x, stats = [], []
                # for each number of parameters
                for l, hyperp_p in enumerate(hyperparams_for_sparse[sparse_from]):
                    # if we have this architecture
                    if j < len(hyperp_p):
                        # what was the density of this hyperp setup?
                        x.append(hyperp_p[j][1])
                        # take the corresponding data
                        stats.append(ar_data_2D[scenario][sparse_from][l][j])
                x, stats = np.array(x), np.array(stats)

            params = hyperp[0][-1]
            neurons = sum(hyperparams_for_sparse[sparse_from][j][-1][0][:-1])

            if j < len(lines):
                # take mean of the means
                ave = np.mean(stats[:, :, stats_i, 0], axis=1)
                mi, ma = get_min_max(stats, stats_i, wide_interval)

                scstr = f'{scenario} ' if len(scenarios) > 1 else ''
                pstr = f'#p {params/1000:.1f}k' if over_neurons else f'#n {neurons}'

                label = f'{scstr}{pstr}'
                if len(stats) > 1:
                    leg1.append(
                        plt.plot(x, ave, label=label,
                                 linestyle=style[subi], color=colors[j+1])[0]
                    )
                    plt.fill_between(x, mi, ma, alpha=0.1, color=colors[j+1])
                else:
                    plt.scatter(x, ave, c=colors[j+1], marker=markers[origi])
                    leg1.append(
                        plt.plot([x, x], [mi, ma],
                                 c=colors[j+1], label=label)[0]
                    )

            if not over_neurons:
                x_params = np.array(
                    parse_densities_from_list_of_hyperparams(hyperp))
                stats_p = ar_data_2D[scenario][sparse_from][j]
                label_params = f'#params {params/1000:.1f}k'

                ave = np.mean(stats_p[:, :, stats_i, 0], axis=1)

                if len(stats_p) > 1:
                    leg2.append(
                        plt.plot(x_params, ave, label=label_params, marker=markers[j], markersize=10,
                                 linestyle=style[-1], color='k')[0]
                    )
                else:
                    leg2.append(
                        plt.scatter(x_params, ave, c='k',
                                    label=label_params, marker=markers[j])
                    )

        bot, top = plt.ylim()
        plt.ylim(bottom=-0.05*top)

        if not just_area:

            first_legend = plt.legend(handles=leg1, loc='upper left')
            plt.gca().add_artist(first_legend)

            if len(leg2) > 0:
                plt.legend(handles=leg2, loc='upper right')
                plt.grid()

            ylabel = '#local 2D ARs' if not normalized_by_area else '#local 2D ARs / area of the local 2D plane'

            wstr = 'wide interval (averages of min & max)' if wide_interval else 'narrow interval (min & max of averages)'
            astr = 'normalized by area' if normalized_by_area else 'absolute numbers'
            trained_str = '' if len(scenarios) > 1 else 'after training'
            plt.title(
                f'Number of local 2D activation regions ({astr}) {trained_str}\naveraged over {average_over_images} 2D planes and {n_models[sparse_from]} models, {wstr}')
        else:
            ylabel = '2D plane area'
            plt.title(
                'Area of the 2 dimensional plane we counted the local ARs on')
            plt.legend()

        plt.ylabel(ylabel)

        if over_neurons:
            plt.xlabel('#neurons')
        else:
            plt.xlabel('density')

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def find_max_ar(ar_data_from_training, sparse_from, wide_interval, stats_i):
    max_ars = 0
    for j, params in enumerate(ar_data_from_training[sparse_from]):
        if wide_interval:
            max_ars = max(max_ars,
                          np.max(
                              ar_data_from_training[sparse_from][params][:, :, :, stats_i])
                          )
        else:
            max_ars = max(max_ars,
                          np.max(
                              ar_data_from_training[sparse_from][params][:, :, :, stats_i, 0])
                          )
        print(max_ars)
    return max_ars


def draw_number_of_2D_ARs_by_iteration(ar_data_from_training,
                                       sparse_from,
                                       normalized_by_area: bool,
                                       use_three_samples,
                                       average_over_images, evaluation_scheme,
                                       hyperparams_for_sparse,
                                       n_models,
                                       xscale='log',
                                       xlim=None,
                                       wide_interval=False,
                                       filename=None, save_figures=False):
    '''
    ar data 2D is Ordered dict key:scenario
        -> OD key:model
            -> [#params]
                -> [#hyperp, #models, #evaluations, #subspaces, normalizedAR/AR/area]
    '''

    n_evals = len(evaluation_scheme)
    cols = 1  # 3
    # int(n_evals / cols) if n_evals % cols == 0 else int(n_evals / cols) + 1
    rows = len(hyperparams_for_sparse[sparse_from])

    style = ['-', '--', '-.', ':']*2
    origi = 1 if use_three_samples else 0

    stats_i = 0
    if not normalized_by_area:
        stats_i = 1

    #y_max = find_max_ar(ar_data_from_training, sparse_from, wide_interval, stats_i)

    def get_min_max(stats, stats_i: int, wide_interval=False):
        assert len(stats.shape) == 4
        if wide_interval:  # averages of min and max
            mi, ma = np.min(stats[:, :, :, stats_i], axis=2), np.max(
                stats[:, :, :, stats_i], axis=2)
            return np.mean(mi, axis=0), np.mean(ma, axis=0)
        else:  # min and max of the averages
            ave = np.mean(stats[:, :, :, stats_i], axis=2)
            return np.min(ave, axis=0), np.max(ave, axis=0)

    for i, hyperp in enumerate(hyperparams_for_sparse[sparse_from]):
        params = hyperp[0][-1]
        fig = plt.figure(figsize=(12, 7), dpi=160)

        for j, stats in enumerate(ar_data_from_training[sparse_from][params]):

            neurons_n = parse_hidden_neuron_count_from_list_of_hyperparams(
                hyperp)

            densities = parse_densities_from_list_of_hyperparams(hyperp)

            stats_averaged_over_subspaces = np.mean(stats, axis=2)

            # average over models
            ave = np.mean(stats_averaged_over_subspaces[:, :, stats_i], axis=0)
            mi, ma = get_min_max(stats, stats_i, wide_interval)

            label = f'#h-neurons {neurons_n[j]}, density {densities[j]:.2f}'
            if len(stats) > 1:
                plt.plot(evaluation_scheme, ave, label=label,
                         linestyle=style[origi], color=f'C{j}')
                plt.fill_between(evaluation_scheme, mi, ma,
                                 alpha=0.1, color=f'C{j}')
            else:
                plt.scatter(evaluation_scheme, ave,
                            c=f'C{j}', marker=markers[origi])
                plt.plot([evaluation_scheme, neurons_n], [mi, ma],
                         c=f'C{j}', label=label)

        plt.ylim(bottom=0)

        plt.legend()  # loc='upper left')

        ylabel = '#local 2D ARs' if not normalized_by_area else '#local 2D ARs / area of the local 2D plane'

        plt.title(f'Number of hidden network parameters {params/1000:.1f}k')
        plt.ylabel(ylabel)
        plt.xscale(xscale)
        xlabel = 'iteration (log)' if 'log' in xscale else 'iteration'
        plt.xlabel(xlabel)
        if xlim is not None:
            plt.xlim(xlim)

        wstr = 'wide interval (averages of min & max)' if wide_interval else 'narrow interval (min & max of averages)'
        astr = 'normalized by area' if normalized_by_area else 'absolute numbers'
        plt.suptitle(f'Local 2D activation regions ({astr}) averaged over {average_over_images} images and {n_models[sparse_from]} models, {wstr}',
                     y=1)

        plt.tight_layout()
        plt.grid()
        fig.patch.set_facecolor('w')

        if filename and save_figures:
            plt.savefig(filename.format(round(params/1000)))
            plt.show()
        else:
            plt.show()


def visualize_subspace_splitting(subspace, filename, save_figures=False):
    plt.figure(figsize=(12, 12))
    subspace.visualize()

    plt.tight_layout()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def visualize_neuron_activation_statuses(subspace, hidden_dims, model_str='', title=None, filename=None, save_figures=False):
    n_n = subspace.n_neurons
    cols = min(int(np.sqrt(n_n)), 10)
    rows = int(n_n / cols)
    rows = rows if rows*cols == n_n else rows + 1
    fig = plt.figure(figsize=(4*cols, 4*rows))

    l = 0
    prev_l_n = 0
    for i in range(rows):
        for j in range(cols):
            n_ind = i*cols+j
            if n_ind == n_n:
                break
            plt.subplot(rows, cols, n_ind+1)

            if hidden_dims[l] - n_ind + prev_l_n < 1:
                prev_l_n += hidden_dims[l]
                l += 1

            n_ind_l = (n_ind-prev_l_n) % hidden_dims[l]
            draw_regions(subspace.regions, title=f'neuron {n_ind_l+1} on layer {l}',
                         color_by_activation_of_neuron_and_layer=(n_ind_l, l))
            plt.axis('equal')

    if model_str is '':
        model_str = f'for {n_n} neurons'

    title = f'Neuron activation statuses {model_str}' if title is None else title
    plt.suptitle(title, y=1)
    plt.tight_layout()
    fig.patch.set_facecolor('w')
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def format_2d_point(point):
    return f'({point[0]:.1f};{point[1]:.1f})'


def visualize_2d_points_as_images(points, base):
    r, c = max(1, int(len(points)) / 4), 4
    plt.figure(figsize=(16, 4*r))
    for j, point in enumerate(points):
        img = project_to_higher_dim(point, base)
        plt.subplot(r, c, j+1)
        plt.imshow(img.reshape(28, 28),
                   cmap='gray')  # , vmin=0, vmax=1)
        plt.title(f'point {format_2d_point(point)}')
        plt.axis('off')


def visualize_region_splitting(model, regions, depth, width, filename=None, save_figures=False,
                               subspace_corners=None, cols=8):
    n_n = depth * width
    rows = n_n * 2 + 1

    if subspace_corners:
        xlim = subspace_corners[0][0], subspace_corners[-1][0]
        ylim = subspace_corners[0][1], subspace_corners[1][1]
    else:
        xlim = ylim = None

    fig = plt.figure(figsize=(cols*4, rows*4))

    l_i = 0

    model_list = list(model.children())
    first = model_list[0]
    if type(first) is torch.nn.ModuleList:
        model_list = first

    prev_layers = []
    for i, layer in enumerate(model_list):
        print(layer)
        if type(layer) is torch.nn.ReLU:
            print('dont mess around with relu')
            continue

        with torch.no_grad():
            weights, biases = extract_bias_and_weights(layer)
            pruning_mask = extract_pruning_mask(layer)
            if pruning_mask is None:
                pruning_mask = torch.ones_like(weights)
            #print(weights.shape, biases.shape, pruning_mask.shape)
            #print(f'bias std {np.std(biases.detach().numpy())}')

            for n_node in range(len(weights)):
                n_ind = l_i*width + n_node
                print('apply node ', n_ind+1)
                print()
                w, b, pm = weights[n_node].detach().numpy(), biases[n_node].detach(
                ).numpy(), pruning_mask[n_node].detach().numpy()
                new_regions = []

                plt.subplot(rows, cols, cols*n_ind*2+1)
                draw_regions(
                    regions, title=f'regions before neuron {n_ind+1} on layer {l_i + 1}')

                print(f'\n### Neuron {n_ind+1}! w {w} b {b} pm {pm}')

                for j, reg in enumerate(regions):
                    print('## reg ', j, end='\r')
                    ind_row = j + 2
                    if ind_row <= cols:
                        plt.subplot(rows, cols, cols*n_ind*2+j+2)
                        draw_regions(
                            [reg], title=f'region {j} before neuron {n_ind+1}',
                            xlim=xlim, ylim=ylim, color_inds_cmap=[j])

                    did_split, regs = reg.update_with_neuron(w, b, prev_layers)

                    regs = regs if type(regs) is tuple else [regs]

                    if ind_row <= cols:
                        plt.subplot(rows, cols, cols*(n_ind*2+1)+j+2)
                        draw_regions(
                            regs, title=f'after split {j+1}', xlim=xlim, ylim=ylim)

                    if did_split:
                        new_regions.append(regs[0])
                        new_regions.append(regs[1])
                    else:
                        new_regions.append(regs[0])
                regions = new_regions

        l_i += 1
        prev_layers.append(layer)

    plt.subplot(rows, cols, cols*(n_ind + 1)*2+1)
    draw_regions(regions, title=f'regions after neuron {n_ind + 1}')
    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()

    return regions

# Neuron horizons


def draw_over_horizon_counts_for_sparse(horizon_stats, sparse_from: str, one_shot_pruning_rates, test_batch_size, hidden_dims_dict, prune_weights: bool, n_models: dict, blacklist=[''], filename=None, xscale='linear', save_figures=False, auto_yscale=False):
    # uaps: key is the scenario, gives OrderedDict key:model -> [#prs, #sparsemodels]
    fig = plt.figure(figsize=(16, 8))
    plt.xscale(xscale)

    multipliers = np.ones_like(
        one_shot_pruning_rates) if prune_weights else one_shot_pruning_rates
    max_hors = [round(hidden_dims_dict[sparse_from][0] * multipliers[ev])
                for ev in range(len(one_shot_pruning_rates))]
    colors = ['y', 'b', 'g', 'r']
    c_dict = {
        'rnd @ init': 0,
        'orig @ init': 1,
        'rnd trained': 2,
        'orig trained': 3,
    }
    plt.gca().invert_xaxis()

    plt.plot(one_shot_pruning_rates, max_hors, c='k', label='max #horizons')

    for i, key in enumerate(horizon_stats.keys()):
        if key in blacklist:
            continue

        hor_stats = horizon_stats[key][sparse_from]
        y_ave = np.mean(hor_stats[:, :, 0], axis=-1)
        y_min = np.min(hor_stats[:, :, 1], axis=-1)
        y_max = np.max(hor_stats[:, :, 2], axis=-1)
        plt.plot(one_shot_pruning_rates, y_ave,
                 alpha=1, label=key, c=colors[c_dict[key]])
        plt.fill_between(one_shot_pruning_rates, y_min,
                         y_max, alpha=.3, color=colors[c_dict[key]])

    if not auto_yscale:
        plt.ylim(0, int(1.02*max_hors[0]))
    plt.title(
        f'Number first layer neuron horizons which average data point crossed\n(max = #first layer neurons) sparse models of {sparse_from}, #models {hor_stats.shape[1]}, #data points {test_batch_size}')
    plt.ylabel('number of horizons datapoints were over')
    plt.xlabel('pruning rate')
    plt.legend()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def aggregate_horizon_distributions_for_many_models(dists, test_batch_size, normalize=True):
    '''
    dists: #models, #neurons
    '''
    dists = dists.astype(int)
    c_models = dists.shape[0]
    bin_counts = np.bincount(np.concatenate(
        dists), minlength=test_batch_size + 1)
    mean = np.mean(dists)
    median = np.median(dists)

    if normalize:
        bin_counts = bin_counts / c_models

    return bin_counts, mean, median


def draw_horizon_distributions_for_sparse(hor_dists, sparse_from: str, one_shot_pruning_rates, test_batch_size, hidden_dims_dict, prune_weights: bool, n_models: dict, blacklist=[''], filename=None, xscale='linear', save_figures=False, density=False, auto_yscale=False, y_max=None, bins=None):
    '''
    hor_dists: key is the scenario, gives OrderedDict key:model -> [#prs, #sparsemodels, #first layer neurons]

    '''
    n_prs = len(one_shot_pruning_rates)
    n_scenarios = len(list(hor_dists.keys())) - len(blacklist)
    fig = plt.figure(figsize=(10 + 4 * n_scenarios, 4*n_prs))
    bins = bins if bins is not None else max(int(test_batch_size / 20), 20)

    colors = ['y', 'b', 'g', 'r']
    c_dict = {
        'rnd @ init': 0,
        'orig @ init': 1,
        'rnd trained': 2,
        'orig trained': 3,
    }

    #legend_strs = [f'model {i+1}' for i in range(n_models[sparse_from])]

    idx_scenario = 0
    for i, key in enumerate(hor_dists.keys()):
        if key in blacklist:
            continue
        idx_scenario += 1

        for j, ms in enumerate(one_shot_pruning_rates):
            ax = plt.subplot(n_prs, n_scenarios, j*n_scenarios + idx_scenario)
            dists = hor_dists[key][sparse_from][j]
            n_neurons = dists.shape[-1]

            dists, mean, median = aggregate_horizon_distributions_for_many_models(
                dists, test_batch_size)

            plt.axvline(mean, color='k')
            plt.axvline(median, color='k', linestyle='--')

            plt.hist(range(test_batch_size + 1), bins=bins, weights=dists,
                     density=density,
                     color=colors[c_dict[key]])

            plt.hist([0], bins=1, weights=[dists[0]],
                     density=density,
                     color='c')

            if not auto_yscale and not density:
                y_top = n_neurons / 3 if y_max is None else y_max
                plt.ylim(0, y_top)
            elif not auto_yscale:
                y_top = 0.1 if y_max is None else y_max
                plt.ylim(0, y_top)

            if i == 0:
                ylabel = '#first layer neurons' if not density else 'density of the #first layer neurons'
                plt.ylabel(ylabel)
            if j == n_prs - 1:
                plt.xlabel('number of data points that made the neuron active')

            title_str = f'{key} - {100*ms:.1f}%'
            legend = [f'mean {mean:.1f}', f'median {median:.1f}',
                      title_str, f'zeros {dists[0]:.1f}']
            plt.legend(legend)
            plt.title(f'{title_str} - #models {n_models[sparse_from]}')

    plt.suptitle(f'Distributions of first layer neurons which cross x horizons for {sparse_from}, #models {n_models[sparse_from]}, bins {bins} => {test_batch_size / bins:.0f} counts per bin',
                 y=1.00)

    plt.tight_layout()
    fig.patch.set_facecolor('w')
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_dark_neuron_movements(dm_diff, one_shot_pruning_rates, n_models, xscale, filename=None, save_figures=False, relative=True):
    fig = plt.figure(figsize=(12, 12))
    c_dict = {
        'rnd0': 'y',
        'orig0': 'b',
        'rnd1': 'g',
        'orig1': 'r',
        'rnd2': 'y',
        'orig2': 'b',
    }
    styles = ['-', '--', '-.']
    relative_str = 'portion' if relative else 'number'
    titles = [
        f'{relative_str} of dark neurons at init',
        f'{relative_str} of dark neurons after training',
        f'{relative_str} of dark neurons which remained dark throughout the training',
    ]

    first_layer_neurons_total = {
        name: hidden_dims_dict[name][0] * n_models[name] for name in model_architectures}

    for k in range(3):
        ax = plt.subplot(3, 1, k+1)
        plt.gca().invert_xaxis()
        plt.xscale(xscale)

        for case in dm_diff:
            for i, name in enumerate(dm_diff[case].keys()):
                if k < 2:
                    normalizer = first_layer_neurons_total[name]
                else:
                    # compare to the number of dark neurons in init
                    normalizer = dm_diff[case][name][:, 1]

                absolute_dark = dm_diff[case][name][:, k]
                relative_dark = absolute_dark / normalizer
                if relative:
                    data = relative_dark
                else:
                    data = absolute_dark

                plt.plot(one_shot_pruning_rates,
                         data,
                         ls=styles[i],
                         label=f'{case}-{name}', color=c_dict[case+str(k)])

        plt.title(titles[k])
        plt.ylabel(f'{relative_str} of neurons')
        plt.xlabel(f'mask sparsity')
        plt.legend()

    plt.suptitle(f'{relative_str} of dark neurons for ' +
                 n_models.__str__(), y=1.00)
    plt.tight_layout()
    fig.patch.set_facecolor('w')
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()

# input features


def draw_input_feature_heatmaps(pruned_feature_dists, sparse_from, one_shot_pruning_rates, input_features, n_models, filename=None, save_figures=False):
    n_ms = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(10, 5*n_ms))
    img_dim = int(np.sqrt(input_features))  # assumes that the image is square
    # assume that 50% of the features are informative
    v_max = min(1, 3 / input_features)
    print(v_max)
    for j, ms in enumerate(one_shot_pruning_rates):
        for i, key in enumerate(pruned_feature_dists.keys()):
            ax = plt.subplot(n_ms, 2, j*2+i+1)
            fl_active = pruned_feature_dists[key][sparse_from][j][:, 0].sum(
                axis=0)
            fl_active_dist = fl_active / fl_active.sum()
            ax.imshow(fl_active_dist.reshape(img_dim, img_dim),
                      cmap='gray', vmin=0, vmax=v_max)
            plt.title(f'{key} with {100*ms:.1f}% of weights remaining')

    plt.suptitle(
        f'Heatmap on which input features (pixels) had active connections', y=1)
    fig.patch.set_facecolor('w')

    plt.tight_layout()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_first_layer_active_connections(pruned_feature_dists, sparse_from, one_shot_pruning_rates, input_features, filename=None, save_figures=False):
    n_ms = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(10, 4*n_ms))
    bin_width = 0.3
    bin_locations = [np.array(
        range(input_features)) + n * bin_width - bin_width / 2 for n in range(2)]

    c_dict = {
        'rnd': 'y',
        'wt': 'b',
    }

    for j, ms in enumerate(one_shot_pruning_rates):
        ax = plt.subplot(n_ms, 1, j+1)
        for i, key in enumerate(pruned_feature_dists.keys()):

            height = pruned_feature_dists[key][sparse_from][j][:, 0].sum(
                axis=0)
            ax.bar(x=bin_locations[i], height=height,
                   width=bin_width, label=key, color=c_dict[key])

        plt.title(
            f'How many neurons have active connection to input feature x_i, sparsity {1-ms:.3f}')
        plt.ylabel('#neurons with unpruned connection to x_i')
        plt.xlabel('input feature x_i')
        plt.legend()
    fig.patch.set_facecolor('w')

    plt.tight_layout()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_information_flow_heatmaps(input_info_through_pruned, sparse_from, one_shot_pruning_rates, input_features, n_models, filename=None, save_figures=False):
    n_ms = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(10, 5*n_ms))
    img_dim = int(np.sqrt(input_features))  # assumes that the image is square
    # assume that 50% of the features are informative
    v_max = min(1, 3 / input_features)
    for j, ms in enumerate(one_shot_pruning_rates):
        for i, key in enumerate(input_info_through_pruned.keys()):
            ax = plt.subplot(n_ms, 2, j*2+i+1)
            input_info = input_info_through_pruned[key][sparse_from][j].sum(
                axis=0)
            input_info /= input_info.sum()
            #print(input_info.max()> v_max)
            ax.imshow(input_info.reshape(img_dim, img_dim),
                      cmap='gray', vmin=0, vmax=v_max)
            plt.title(f'{key} with {100*(ms):.1f}% of weights remaining')
            plt.axis('off')

    plt.suptitle(
        f'Heatmap on which input features were listened by the last pruned layer', y=1)
    fig.patch.set_facecolor('w')

    plt.tight_layout()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_input_info_passed_through_pruned(input_info_through_pruned, sparse_from, one_shot_pruning_rates, input_features, filename=None, save_figures=False):
    n_ms = len(one_shot_pruning_rates)
    fig = plt.figure(figsize=(10, 4*n_ms))
    bin_width = 0.3
    bin_locations = [np.array(
        range(input_features)) + n * bin_width - bin_width / 2 for n in range(2)]

    c_dict = {
        'rnd': 'y',
        'wt': 'b',
    }

    for j, ms in enumerate(one_shot_pruning_rates):
        ax = plt.subplot(n_ms, 1, j+1)
        for i, key in enumerate(input_info_through_pruned.keys()):

            height = input_info_through_pruned[key][sparse_from][j].sum(
                axis=0)
            height /= height.sum()
            ax.bar(x=bin_locations[i], height=height,
                   width=bin_width, label=key, color=c_dict[key])

        plt.title(
            f'How much input features were available later in the network? Sparsity {1-ms:.3f}')
        plt.ylabel('density how much feature x_i was "listened"')
        plt.xlabel('input feature x_i')
        plt.legend()
    fig.patch.set_facecolor('w')

    plt.tight_layout()
    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


# Decision trees

def draw_tree_data_constant_param(tree_data, metrics, sparse_from, over_neurons,
                                  param_keys, lp_samples, hyperparams_for_sparse,
                                  arc_layers, metrics_dict,
                                  draw_init=True, xscale='logit', filename=None, save_figures=False):
    '''
    tree data is OD key:scenario
        -> OD key:architecture
            -> [#params]
                -> [#hyperp, #metrics, ave/mi/ma]
    '''

    n_metrics = len(metrics)

    fig = plt.figure(figsize=(20, 6*n_metrics))

    metrics_to_draw = [metrics_dict[metric] for metric in metrics]

    style = ['--', '-', '-.']

    if over_neurons:
        lines = param_keys[sparse_from]  # lines share the parameter count
    else:  # lines share the neuron count
        lines = parse_hidden_neuron_count_from_list_of_hyperparams(
            hyperparams_for_sparse[sparse_from][-1])

    for k, metric in enumerate(metrics):
        plt.subplot(n_metrics, 1, k+1)

        leg1_handles, leg2_handles = [], []

        for i, scenario in enumerate(list(tree_data.keys())):

            if 'init' in scenario and not draw_init:
                continue

            for j, hyperp in enumerate(hyperparams_for_sparse[sparse_from]):
                # what should we have on the x-axis?
                if over_neurons:
                    x = parse_hidden_neuron_count_from_list_of_hyperparams(
                        hyperp)
                    tree_d = np.array(tree_data[scenario][sparse_from][j])
                else:
                    # draw the lines between models with the same amount of neurons
                    x, tree_d = [], []
                    # for each number of parameters
                    for l, hyperp_p in enumerate(hyperparams_for_sparse[sparse_from]):
                        if j < len(hyperp_p):
                            # what was the sparsity of this hyperp setup?
                            x.append(hyperp_p[j][1])
                            # take the corresponding data
                            tree_d.append(
                                tree_data[scenario][sparse_from][l][j])
                    x, tree_d = np.array(x), np.array(tree_d)

                if j < len(lines):
                    ave, mi, ma = [tree_d[:, metrics_to_draw[k], n]
                                   for n in range(3)]

                    label = f'{scenario} #p {lines[j]/1000:.1f}k' if over_neurons else f'{scenario} #n {lines[j]}'

                    if len(tree_d) > 1:
                        leg1_handles.append(
                            plt.plot(x, ave, label=label,
                                     linestyle=style[i], color=colors[j+1])[0]
                        )
                        plt.fill_between(x, mi, ma, alpha=0.1,
                                         color=colors[j+1])
                    else:
                        plt.scatter(x, ave, c=colors[j+1], marker=markers[i])
                        leg1_handles.append(
                            plt.plot([x, x], [mi, ma], c=colors[j+1],
                                     label=label, alpha=0.8)[0]
                        )

                if not over_neurons:
                    # data to draw lines between the models with same number of params
                    params = param_keys[sparse_from][j]
                    x_params = parse_densities_from_list_of_hyperparams(hyperp)
                    tree_d_params = np.array(
                        tree_data[scenario][sparse_from][j])
                    label_params = f'#params {params/1000:.1f}k'

                    ave, mi, ma = [tree_d_params[:, metrics_to_draw[k], n]
                                   for n in range(3)]

                    if len(tree_d_params) > 1:
                        leg2_handles.append(
                            plt.plot(x_params, ave, label=label_params, marker=markers[j], markersize=10,
                                     linestyle=style[-1], color='k')[0]
                        )
                    else:
                        leg2_handles.append(
                            plt.scatter(x_params, ave, c='k',
                                        label=label_params, marker=markers[j])
                        )

        if over_neurons:
            plt.xlabel('#neurons')
        else:
            plt.xlabel('density')

        first_legend = plt.legend(handles=leg1_handles, loc='upper left')
        plt.gca().add_artist(first_legend)

        if len(leg2_handles) > 0:
            plt.legend(handles=leg2_handles, loc='lower left')

        plt.title(metric)

    plt.suptitle(f'Statistics from decision tree trained with {lp_samples} samples of {sparse_from} hidden layer {arc_layers[sparse_from] + 1} layer patterns',
                 y=1)

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_tree_data(tree_data, metrics, sparse_from, one_shot_pruning_rates, lp_samples, arc_layers, metrics_dict, xscale='logit', filename=None, save_figures=False):
    n_metrics = len(metrics)

    fig = plt.figure(figsize=(20, 6*n_metrics))

    metrics_to_draw = [metrics_dict[metric] for metric in metrics]

    c_dict = {
        'LT init': 'y',
        'LT trained': 'g',
        'WT init': 'b',
        'WT trained': 'r'
    }

    for j, metric in enumerate(metrics):
        plt.subplot(n_metrics, 1, j+1)
        plt.xscale(xscale)
        plt.gca().invert_xaxis()

        for i, scenario in enumerate(list(tree_data.keys())):
            ave, mi, ma = [tree_data[scenario][sparse_from]
                           [:, metrics_to_draw[j], k] for k in range(3)]
            plt.plot(one_shot_pruning_rates, ave,
                     c=c_dict[scenario], label=scenario)
            plt.fill_between(one_shot_pruning_rates, mi, ma,
                             alpha=.05, color=c_dict[scenario])

        plt.legend()
        plt.title(metric)

    plt.suptitle(f'Statistics from decision tree trained with {lp_samples} samples of {sparse_from} hidden layer {arc_layers[sparse_from] + 1} layer patterns',
                 y=1)

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_tree_width_by_layer(level_sizes, leaves_per_level, ave_depth):
    decision_nodes_lvl = level_sizes - leaves_per_level

    fig = plt.figure(figsize=(12, 5))
    plt.plot(range(tree_depth+1), level_sizes, label='nodes per lvl')
    plt.plot(range(tree_depth+1), leaves_per_level, label='leaves per lvl')
    plt.plot(range(tree_depth+1), decision_nodes_lvl,
             label='decision nodes lvl')
    plt.axvline(ave_depth, label=f'average depth {ave_depth:.1f}', c='k')

    plt.xlabel('level')
    plt.ylabel('#dtree nodes')
    plt.legend()
    plt.show()


def draw_hyperparam_setup(hyperparams_for_sparse, filename=None, save_figures=False):
    colors = ['k', 'c', 'm', 'r', 'g', 'b', 'y'] * 10

    n_arc = len(list(hyperparams_for_sparse.keys()))
    fig = plt.figure(figsize=(12, n_arc*6))

    for i, sparse_from in enumerate(hyperparams_for_sparse):
        plt.subplot(n_arc, 1, i+1)
        for j, hyp_p in enumerate(hyperparams_for_sparse[sparse_from]):
            nets = np.array([
                [hyp[1], sum(hyp[0][:-1])] for hyp in hyp_p
            ])
            plt.plot(nets[:, 0], nets[:, 1], marker=markers[j], ms=12,
                     c=colors[i], label=f'n_params = {hyp_p[0][2]/1000:.1f}k')

        plt.title(
            f'Networks with the same number of active parameters for {sparse_from}')
        plt.ylabel('#neurons')
        plt.legend()
    plt.xlabel('density')
    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


# specialization
def get_spec_filename_end(sparse_from, wide_interval, draw_init, over_neurons, class_cov_i):
    distr = '_with-init' if draw_init else ''
    wistr = 'wide' if wide_interval else 'narrow'
    orstr = 'over-neurons' if over_neurons else 'over-density'
    ccstr = f'_cc{round(class_coverage[class_cov_i]*1000)}' if class_cov_i else ''
    return f'{sparse_from}_{orstr}_{wistr}{distr}{ccstr}.png'


def draw_specialization_data(specialization_data, metrics,
                             sparse_from, over_neurons,
                             param_keys, lp_samples, hyperparams_for_sparse,
                             wide_interval,
                             arc_layers, metrics_dict,
                             average_over_images,
                             use_three_samples,
                             class_cov_i=None,
                             class_coverages=None,
                             draw_init=True, xscale='logit',
                             filename=None, save_figures=False):
    '''

    specialization data is OD key:scenario
        -> OD key:architecture
            -> [#params]
                -> [#hyperp, #models, #subspaces, #coverages, #metrics]

    '''

    assert class_cov_i is None or type(
        class_cov_i) == int, 'which class coverage index should be int'
    assert class_coverages is None or type(
        class_coverages) == list, 'class coverages should be a list'

    n_metrics = len(metrics)

    if n_metrics > 1:
        fig = plt.figure(figsize=(12, 4*n_metrics), dpi=160)
    else:
        fig = plt.figure(figsize=(8, 6), dpi=160)

    metrics_to_draw = [metrics_dict[metric] for metric in metrics]

    scenarios = list(specialization_data.keys())

    style = ['--', '-', '-.'] if len(scenarios) > 1 else ['-', '-.']

    if over_neurons:
        lines = param_keys[sparse_from]  # lines share the parameter count
    else:  # lines share the neuron count
        lines = parse_hidden_neuron_count_from_list_of_hyperparams(
            hyperparams_for_sparse[sparse_from][-1])

    def get_min_max(stats, stats_i, wide_interval):
        if wide_interval:  # averages of subspace min and max
            mi, ma = np.min(stats[:, :, :, stats_i], axis=2), np.max(
                stats[:, :, :, stats_i], axis=2)
            return np.mean(mi, axis=1), np.mean(ma, axis=1)
        else:  # min and max of the subspace averages
            ave = np.mean(stats[:, :, :, stats_i], axis=2)
            return np.min(ave, axis=1), np.max(ave, axis=1)

    for k, metric in enumerate(metrics):
        plt.subplot(n_metrics, 1, k+1)

        leg1_handles, leg2_handles = [], []

        for i, scenario in enumerate(scenarios):

            if 'init' in scenario and not draw_init:
                continue

            for j, hyperp in enumerate(hyperparams_for_sparse[sparse_from]):

                # ----------------- prep -------------------

                # what should we have on the x-axis?
                if over_neurons:
                    x = np.array(parse_hidden_neuron_count_from_list_of_hyperparams(
                        hyperp))
                    # average over subspaces
                    spec_d = np.array(
                        specialization_data[scenario][sparse_from][j][:,
                                                                      :, :, class_cov_i, :]
                    )
                else:
                    # draw the lines between models with the same amount of neurons
                    # we need to organize the data differently
                    x, spec_d = [], []
                    # for each number of parameters
                    for l, hyperp_p in enumerate(hyperparams_for_sparse[sparse_from]):
                        # if we have this architecture
                        if j < len(hyperp_p):
                            # what was the sparsity of this hyperp setup?
                            x.append(hyperp_p[j][1])
                            # take the corresponding data
                            # average over subspaces
                            spec_d.append(
                                specialization_data[scenario][sparse_from][l][j][:,
                                                                                 :, class_cov_i, :]
                            )

                    x, spec_d = np.array(x), np.array(spec_d)

                # -------------------------- plot the colourful lines ---------------------------

                if j < len(lines):
                    # average over susbpace aves
                    spec_d_subsave = np.mean(
                        spec_d[:, :, :, metrics_to_draw[k]], axis=2)
                    # average over model averages
                    ave = np.mean(spec_d_subsave, axis=1)

                    assert ave.shape[0] == x.shape[
                        0], f'shapes were incompatible, {ave.shape} {x.shape}'

                    mi, ma = get_min_max(
                        spec_d, metrics_to_draw[k], wide_interval)

                    label = f'{scenario} #p {lines[j]/1000:.1f}k' if over_neurons else f'{scenario} #n {lines[j]}'

                    if len(spec_d) > 1:
                        leg1_handles.append(
                            plt.plot(x, ave, label=label,
                                     linestyle=style[i], color=colors[j+1])[0]
                        )
                        plt.fill_between(x, mi, ma, alpha=0.1,
                                         color=colors[j+1])
                    else:
                        leg1_handles.append(
                            plt.scatter(
                                x, ave, c=colors[j+1], label=label, marker=markers[i])
                        )
                        plt.plot([x, x], [mi, ma], c=colors[j+1], alpha=0.5)

                # ------------------- plot the dashed black lines -------------------------

                if not over_neurons and not draw_init:
                    # data to draw lines between the models with same number of params
                    params = param_keys[sparse_from][j]
                    x_params = np.array(
                        parse_densities_from_list_of_hyperparams(hyperp))

                    spec_d_params = np.array(
                        specialization_data[scenario][sparse_from][j][:,
                                                                      :, :, class_cov_i, :]
                    )
                    label_params = f'#params {params/1000:.1f}k'

                    # average over models and subspaces
                    ave = np.mean(
                        spec_d_params[:, :, :, metrics_to_draw[k]], axis=2)
                    ave = np.mean(ave, axis=1)

                    assert ave.shape[0] == x_params.shape[
                        0], f'shapes were incompatible, {ave.shape} {x_params.shape}'

                    mi, ma = get_min_max(
                        spec_d_params, metrics_to_draw[k], wide_interval)

                    if len(spec_d_params) > 1:
                        leg2_handles.append(
                            plt.plot(x_params, ave, label=label_params, marker=markers[j], markersize=10,
                                     linestyle=style[-1], color='k')[0]
                        )
                    else:
                        leg2_handles.append(
                            plt.scatter(x_params, ave, c='k',
                                        label=label_params, marker=markers[j])
                        )

        if over_neurons:
            plt.xlabel('#neurons')
        else:
            plt.xlabel('density')

        # specialization should start from 0
        if 'specialization' in metric:
            plt.ylabel('specialization $s$')
            if n_metrics == 1:
                b, t = plt.ylim(bottom=0)
                plt.yticks(np.arange(0, t, 0.1))
            else:
                plt.ylim(0, 1.02)

        # specialization has different loc
        if 'specialization' in metric and not over_neurons:
            y = 0.15 + n_metrics * 0.2 * 0.15
            first_loc = (0.005, y)
            first_loc = 'lower right'
        else:
            first_loc = 'lower right'
        first_legend = plt.legend(handles=leg1_handles, loc=first_loc)
        plt.gca().add_artist(first_legend)

        if len(leg2_handles) > 0:
            plt.legend(handles=leg2_handles, loc='lower left')

        plt.grid(True)
        plt.title(metric)

    utstr = 'three images' if use_three_samples else 'origin and two images'
    ccstr = f' minimum coverage {class_coverages[class_cov_i]},' if class_coverages else ','
    if n_metrics > 1:
        plt.suptitle(f'{sparse_from}, #samples {lp_samples}{ccstr} subpatterns projected over {average_over_images} different subspaces spun by {utstr}',
                     y=1)
    elif 'specialization' in metric:
        plt.suptitle(f'{sparse_from}{ccstr} averaged over {average_over_images} different subspaces spun by {utstr}',
                     y=1)

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
        plt.show()
    else:
        plt.show()


def draw_aggregated_specialization_data(specialization_data, metrics,
                                        sparse_from, over_neurons,
                                        param_keys, lp_samples, hyperparams_for_sparse,
                                        wide_interval,
                                        arc_layers, metrics_dict,
                                        average_over_images,
                                        use_three_samples,
                                        class_cov_i=None,
                                        class_coverages=None,
                                        draw_init=True, xscale='logit',
                                        filename=None, save_figures=False):
    '''
    specialization data is OD key:scenario
        -> OD key:architecture
            -> [#params]
                -> [#hyperp, #models, #metrics, ave/mi/ma]

    if class_cov_i is None, then there has been only one class coverage (old struct), and 
        the data struct of specialization data has one dim less.
    '''

    assert class_cov_i is None or type(
        class_cov_i) == int, 'which class coverage index should be int'
    assert class_coverages is None or type(
        class_coverages) == list, 'class coverages should be a list'

    n_metrics = len(metrics)

    if n_metrics > 1:
        fig = plt.figure(figsize=(12, 4*n_metrics), dpi=160)
    else:
        fig = plt.figure(figsize=(8, 6), dpi=160)

    metrics_to_draw = [metrics_dict[metric] for metric in metrics]

    scenarios = list(specialization_data.keys())

    style = ['--', '-', '-.'] if len(scenarios) > 1 else ['-', '-.']

    if over_neurons:
        lines = param_keys[sparse_from]  # lines share the parameter count
    else:  # lines share the neuron count
        lines = parse_hidden_neuron_count_from_list_of_hyperparams(
            hyperparams_for_sparse[sparse_from][-1])

    def get_min_max(stats, stats_i, wide_interval):
        '''expects that stats does have only one cc, i.e. that dim is reduced if it did exist'''
        if wide_interval:  # averages of min and max
            return np.mean(stats[:, :, stats_i, 1], axis=1), np.mean(stats[:, :, stats_i, 2], axis=1)
        else:  # min and max of the averages
            return np.min(stats[:, :, stats_i, 0], axis=1), np.max(stats[:, :, stats_i, 0], axis=1)

    for k, metric in enumerate(metrics):
        plt.subplot(n_metrics, 1, k+1)

        leg1_handles, leg2_handles = [], []

        for i, scenario in enumerate(scenarios):

            if 'init' in scenario and not draw_init:
                continue

            for j, hyperp in enumerate(hyperparams_for_sparse[sparse_from]):

                # what should we have on the x-axis?
                if over_neurons:
                    x = np.array(parse_hidden_neuron_count_from_list_of_hyperparams(
                        hyperp))
                    if class_cov_i is not None:  # new struct
                        spec_d = np.array(
                            specialization_data[scenario][sparse_from][j][:, :, :, class_cov_i, :])
                    else:  # old struct
                        spec_d = np.array(
                            specialization_data[scenario][sparse_from][j])

                else:
                    # draw the lines between models with the same amount of neurons
                    # we need to organize the data differently
                    x, spec_d = [], []
                    # for each number of parameters
                    for l, hyperp_p in enumerate(hyperparams_for_sparse[sparse_from]):
                        # if we have this architecture
                        if j < len(hyperp_p):
                            # what was the sparsity of this hyperp setup?
                            x.append(hyperp_p[j][1])
                            # take the corresponding data
                            if class_cov_i is not None:
                                spec_d.append(
                                    specialization_data[scenario][sparse_from][l][j][:, :, class_cov_i, :])
                            else:
                                spec_d.append(
                                    specialization_data[scenario][sparse_from][l][j])
                    x, spec_d = np.array(x), np.array(spec_d)

                if j < len(lines):
                    # average over model averages
                    ave = np.mean(spec_d[:, :, metrics_to_draw[k], 0], axis=1)

                    assert ave.shape[0] == x.shape[
                        0], f'shapes were incompatible, {ave.shape} {x.shape}'

                    mi, ma = get_min_max(
                        spec_d, metrics_to_draw[k], wide_interval)

                    label = f'{scenario} #p {lines[j]/1000:.1f}k' if over_neurons else f'{scenario} #n {lines[j]}'

                    if len(spec_d) > 1:
                        leg1_handles.append(
                            plt.plot(x, ave, label=label,
                                     linestyle=style[i], color=colors[j+1])[0]
                        )
                        plt.fill_between(x, mi, ma, alpha=0.1,
                                         color=colors[j+1])
                    else:
                        leg1_handles.append(
                            plt.scatter(
                                x, ave, c=colors[j+1], label=label, marker=markers[i])
                        )
                        plt.plot([x, x], [mi, ma], c=colors[j+1], alpha=0.5)

                if not over_neurons and not draw_init:
                    # data to draw lines between the models with same number of params
                    params = param_keys[sparse_from][j]
                    x_params = np.array(
                        parse_densities_from_list_of_hyperparams(hyperp))

                    if class_cov_i is not None:
                        spec_d_params = np.array(
                            specialization_data[scenario][sparse_from][j][:, :, :, class_cov_i, :])
                    else:
                        spec_d_params = np.array(
                            specialization_data[scenario][sparse_from][j])

                    label_params = f'#params {params/1000:.1f}k'

                    # average over model averages
                    ave = np.mean(
                        spec_d_params[:, :, metrics_to_draw[k], 0], axis=1)

                    assert ave.shape[0] == x_params.shape[
                        0], f'shapes were incompatible, {ave.shape} {x_params.shape}'

                    mi, ma = get_min_max(
                        spec_d_params, metrics_to_draw[k], wide_interval)

                    if len(spec_d_params) > 1:
                        leg2_handles.append(
                            plt.plot(x_params, ave, label=label_params, marker=markers[j], markersize=10,
                                     linestyle=style[-1], color='k')[0]
                        )
                    else:
                        leg2_handles.append(
                            plt.scatter(x_params, ave, c='k',
                                        label=label_params, marker=markers[j])
                        )

        if over_neurons:
            plt.xlabel('#neurons')
        else:
            plt.xlabel('density')

        # specialization should start from 0
        if 'specialization' in metric:
            plt.ylabel('specialization $s$')
            if n_metrics == 1:
                b, t = plt.ylim(bottom=0)
                plt.yticks(np.arange(0, t, 0.1))
            else:
                plt.ylim(0, 1.02)

        # specialization has different loc
        if 'specialization' in metric and not over_neurons:
            y = 0.15 + n_metrics * 0.2 * 0.15
            first_loc = (0.005, y)
            first_loc = 'lower right'
        else:
            first_loc = 'lower right'
        first_legend = plt.legend(handles=leg1_handles, loc=first_loc)
        plt.gca().add_artist(first_legend)

        if len(leg2_handles) > 0:
            plt.legend(handles=leg2_handles, loc='lower left')

        plt.grid(True)
        plt.title(metric)

    utstr = 'three images' if use_three_samples else 'origin and two images'
    ccstr = f' minimum coverage {class_coverages[class_cov_i]},' if class_coverages else ','
    if n_metrics > 1:
        plt.suptitle(f'{sparse_from}, #samples {lp_samples}{ccstr} subpatterns projected over {average_over_images} different subspaces spun by {utstr}',
                     y=1)
    elif 'specialization' in metric:
        plt.suptitle(f'{sparse_from}{ccstr} averaged over {average_over_images} different subspaces spun by {utstr}',
                     y=1)

    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def draw_specialization_data_over_min_coverage(specialization_data,
                                               name,
                                               class_coverage,
                                               param_keys,
                                               lp_samples,
                                               hyperparams,
                                               average_over_images,
                                               use_three_samples,
                                               draw_init=False,
                                               xscale='linear',
                                               scenario='trained',
                                               linestyles=[
                                                   '-', '--', '-.', ':'],
                                               filename=None,
                                               save_figures=False):
    '''

    specialization data is OD key:scenario
        -> OD key:architecture
            -> [#params]
                -> [#hyperp, #models, #subspaces, #coverages, #metrics]
    '''
    scenarios = [scenario]
    if draw_init:
        scenarios.append('init')

    for i, params in enumerate(param_keys[name]):
        fig = plt.figure(figsize=(12, 7), dpi=160)
        for k, scenario in enumerate(scenarios):
            for j, spec_h in enumerate(specialization_data[scenario][name][i]):
                hyperp = hyperparams[name][i]
                neurons = parse_hidden_neuron_count_from_list_of_hyperparams(hyperp)[
                    j]
                density = parse_densities_from_list_of_hyperparams(hyperp)[j]

                ave_subs = np.mean(spec_h[:, :, :, 0], axis=1)
                ave, mi, ma = (
                    np.mean(ave_subs, axis=0),
                    np.min(ave_subs, axis=0),
                    np.max(ave_subs, axis=0),
                )

                plt.title(
                    f'number of hidden network parameters {params/1000:.1f}k')
                label = f'#h-neurons {neurons}, density {density:.2f}'
                # linestyle=linestyles[i],
                plt.plot(class_coverage, ave,
                         color=f'C{j}', linestyle=linestyles[k], label=label)
                plt.fill_between(class_coverage, mi, ma,
                                 color=f'C{j}', alpha=0.1)

        plt.ylabel('specialization measure $s$')
        plt.xlabel('minimum coverage $c_P$')
        plt.legend()
        plt.grid(zorder=-5)
        plt.ylim((0, 1))

        plt.tight_layout()
        fig.patch.set_facecolor('w')

        if filename and save_figures:
            plt.savefig(filename.format(f'{params/1000:.0f}'))
        else:
            plt.show()


def draw_aggregated_specialization_data_over_min_coverage(specialization_data,
                                                          name,
                                                          class_coverage,
                                                          param_keys,
                                                          lp_samples,
                                                          hyperparams,
                                                          average_over_images,
                                                          use_three_samples,
                                                          draw_init=False,
                                                          xscale='linear',
                                                          scenario='trained',
                                                          linestyles=[
                                                              '-', '--', '-.', ':'],
                                                          filename=None,
                                                          save_figures=False):

    scenarios = [scenario]
    if draw_init:
        scenarios.append('init')

    for i, params in enumerate(param_keys[name]):
        fig = plt.figure(figsize=(12, 7), dpi=160)
        for k, scenario in enumerate(scenarios):
            for j, spec_h in enumerate(specialization_data[scenario][name][i]):
                hyperp = hyperparams[name][i]
                neurons = parse_hidden_neuron_count_from_list_of_hyperparams(hyperp)[
                    j]
                density = parse_densities_from_list_of_hyperparams(hyperp)[j]
                ave, mi, ma = (
                    np.mean(spec_h[:, 0, :, i], axis=0) for i in range(3)
                )

                plt.title(
                    f'number of hidden network parameters {params/1000:.1f}k')
                label = f'#h-neurons {neurons}, density {density:.2f}'
                # linestyle=linestyles[i],
                plt.plot(class_coverage, ave,
                         color=f'C{j}', linestyle=linestyles[k], label=label)
                plt.fill_between(class_coverage, mi, ma,
                                 color=f'C{j}', alpha=0.1)

        plt.ylabel('specialization measure $s$')
        plt.xlabel('minimum coverage $c_P$')
        plt.legend()
        plt.grid(zorder=-5)
        plt.ylim((0, 1))

        plt.tight_layout()
        fig.patch.set_facecolor('w')

        if filename and save_figures:
            plt.savefig(filename.format(f'{params/1000:.0f}'))
        else:
            plt.show()


def draw_specialization_from_training(specialization_data,
                                      name,
                                      hyperparams,
                                      cov_i: list,
                                      class_coverage,
                                      xscale,
                                      evaluation_scheme,
                                      xlim=None,
                                      filename=None,
                                      save_figures=False):

    linestyles = ['-', '--', '-.', ':']*2

    many_covs = len(cov_i) > 1

    for i, hyperp in enumerate(hyperparams[name]):
        fig = plt.figure(figsize=(12, 7), dpi=160)
        labels1 = []
        params = hyperp[0][-1]
        for j, spec_h in enumerate(specialization_data[name][params]):
            neurons = parse_hidden_neuron_count_from_list_of_hyperparams(hyperp)[
                j]
            density = parse_densities_from_list_of_hyperparams(hyperp)[j]

            label = f'#h-neurons {neurons}, density {density:.2f}'
            labels1.append(label)

            for k, c_i in enumerate(cov_i):
                # spec_h.shape = [#models, #evaluations, #subspaces, #coverages, #metrics], e.g., (3, 28, 5, 4, 23)
                # the first metric is the specialization measure
                spec_i = 0
                spec_averaged_over_subspaces = np.mean(
                    spec_h[:, :, :, c_i, spec_i], axis=2)

                ave_over_models, mi, ma = (
                    np.mean(spec_averaged_over_subspaces, axis=0),
                    np.min(spec_averaged_over_subspaces, axis=0),
                    np.max(spec_averaged_over_subspaces, axis=0),
                )

                plt.plot(evaluation_scheme, ave_over_models,
                         color=f'C{j}', linestyle=linestyles[k], label=label)
                plt.fill_between(evaluation_scheme, mi, ma,
                                 color=f'C{j}', alpha=0.1)

        title = f'hidden network parameters {params/1000:0.1f}k'

        plt.title(title)
        plt.ylabel('specialization $s$')
        xlabel = 'iteration (log)' if 'log' in xscale else 'iteration'
        plt.xlabel(xlabel)
        plt.xscale(xscale)
        plt.ylim(0, 1)
        if xlim is not None:
            plt.xlim(xlim)

        # legends
        leg1, leg2 = [], []
        for i in range(len(hyperp)):
            leg1.append(plt.fill_between([0], 0, color=f'C{i}'))

        labels2 = []
        for i in range(len(cov_i)):
            leg2.append(plt.plot(0, 0, color='k', linestyle=linestyles[i])[0])
            labels2.append(f'$c_P = {class_coverage[cov_i[i]]}$')

        plt.gca().add_artist(plt.legend(leg1, labels1, loc='upper left'))
        plt.legend(leg2, labels2, loc=(
            0.006, 0.94-len(leg1)*0.03-len(cov_i)*0.03))

        plt.grid()
        plt.tight_layout()

        if filename and save_figures:
            plt.savefig(filename.format(f'{params/1000:.0f}'))
            plt.show()
        else:
            plt.show()
