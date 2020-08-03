from sklearn import tree
import numpy as np

from .activation_patterns import parse_layers_patterns_to_numpy, get_samples_wrt_postcondition


def get_tree_structure(dtree):
    # from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    n_nodes = dtree.tree_.node_count
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    return n_nodes, node_depth, is_leaves


def get_average_support(n_leaves, lp_samples):
    average_support = lp_samples / n_leaves
    return average_support


def get_nodes_per_level(tree_depth, node_depth):
    return np.bincount(node_depth, minlength=tree_depth+1)


def get_leaf_depths(node_depth, is_leaves):
    return node_depth[is_leaves]


def get_average_depth(leaf_depth):
    return np.mean(leaf_depth)


def get_tree_info(dtree):
    # decision tree ready made
    tree_depth = dtree.get_depth()
    n_leaves = dtree.get_n_leaves()

    # custom
    n_nodes, node_depth, is_leaves = get_tree_structure(dtree)
    leaf_depth = get_leaf_depths(node_depth, is_leaves)
    ave_depth = get_average_depth(leaf_depth)
    level_sizes = get_nodes_per_level(tree_depth, node_depth)
    leaves_per_lvl = get_nodes_per_level(tree_depth, leaf_depth)

    return tree_depth, n_leaves, level_sizes, leaves_per_lvl, ave_depth


def get_useful_neurons(dtree):
    # How many activation statuses were useful for the dtree?
    feature = dtree.tree_.feature
    uvalues, ucounts = np.unique(feature, return_counts=True)
    uvalues, ucounts = uvalues[1:], ucounts[1:]  # discard leaves

    # how many neurons (features) were used in the tree?
    useful_neurons = len(uvalues)
    return useful_neurons, uvalues, ucounts


def get_leaves_with_support_for_class(dtree, patterns, targets, c: int):
    mask = targets == c
    n_c = mask.sum()
    if n_c == 0:  # there is no sample from this class
        return np.array([[-1, 0]])
    samples_pc = patterns[mask]  # the samples that satisfy the postcondition
    leaf_ids = dtree.apply(samples_pc)
    uleaves, uleaves_c = np.unique(leaf_ids, return_counts=True)
    # normalize wrt the number of samples in this class to make classes comparable with each other
    uleaves_c = uleaves_c / n_c
    return np.array((uleaves, uleaves_c)).T


def get_max_support(uleaves_class_c):
    idx = uleaves_class_c[:, 1].argmax()
    return uleaves_class_c[idx][1]


def get_number_leaves_for_class(uleaves_class_c):
    return len(uleaves_class_c)


def aggregate_trees_over_models(tree_data_models):
    return np.array([
        np.mean(tree_data_models, axis=0),
        np.min(tree_data_models, axis=0),
        np.max(tree_data_models, axis=0)
    ]).T


def tree_data_for_model(lpi, targets):
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(lpi, targets)

    # info about the decision tree
    tree_depth, n_leaves, level_sizes, leaves_per_level, ave_depth = get_tree_info(
        dtree)

    # how many neuron's activation were observed?
    useful_neurons, _, _ = get_useful_neurons(dtree)

    # what was the maximum support for a pattern for each class?
    leaves_with_support = [
        get_leaves_with_support_for_class(dtree, lpi, targets,
                                          c=c)
        for c in range(10)
    ]
    max_support_per_class = [
        get_max_support(uleaves) for uleaves in leaves_with_support
    ]

    # how many patterns there were per class
    n_patterns_class = [
        get_number_leaves_for_class(uleaves) for uleaves in leaves_with_support
    ]

    return [tree_depth, n_leaves, ave_depth, useful_neurons] + max_support_per_class + n_patterns_class


def tree_data_for_layer_i_constant_param(lps_arc, targets_arc, sparse_from, layer_i, param_keys):
    lp_data = lps_arc[sparse_from]
    target_data = targets_arc[sparse_from]

    tree_param = []
    for j, params in enumerate(param_keys[sparse_from]):
        tree_h = []

        for i in range(len(lp_data[j])):

            tree_a = []
            for k, patterns in enumerate(lp_data[j][i]):

                lpi = parse_layers_patterns_to_numpy(patterns, layer_i)
                targets = target_data[j][i][k]

                tree_a.append(tree_data_for_model(lpi, targets))

            tree_h.append(aggregate_trees_over_models(np.array(tree_a)))

        tree_param.append(np.array(tree_h))

    return np.array(tree_param)


def tree_data_for_layer_i(lps_arc, targets_arc, sparse_from, layer_i, one_shot_pruning_rates):
    '''
    LT vs WT experiment setup
    '''

    lp_data = lps_arc[sparse_from]
    target_data = targets_arc[sparse_from]

    tree_ms = []
    for i, ms in enumerate(one_shot_pruning_rates):
        tree_a = []
        for j, patterns in enumerate(lp_data[i]):
            lpi = parse_layers_patterns_to_numpy(patterns, layer_i)
            targets = target_data[i, j]

            dtree = tree.DecisionTreeClassifier()
            dtree.fit(lpi, targets)

            # info about the decision tree
            tree_depth, n_leaves, level_sizes, leaves_per_level, ave_depth = get_tree_info(
                dtree)

            # how many neuron's activation were observed?
            useful_neurons, _, _ = get_useful_neurons(dtree)

            # what was the maximum support for a pattern for each class?
            leaves_with_support = [
                get_leaves_with_support_for_class(dtree, lpi, targets,
                                                  c=c)
                for c in range(10)
            ]
            max_support_per_class = [
                get_max_support(uleaves) for uleaves in leaves_with_support
            ]
            n_patterns_class = [
                get_number_leaves_for_class(uleaves) for uleaves in leaves_with_support
            ]

            tree_a.append([tree_depth, n_leaves, ave_depth, useful_neurons] +
                          max_support_per_class + n_patterns_class)

        tree_ms.append(aggregate_trees_over_models(np.array(tree_a)))

    return np.array(tree_ms)


def remove_leaf_nodes(dtree_nodes, is_leaves):
    '''
    returns a numpy array with leave nodes removed
    '''
    return np.setdiff1d(dtree_nodes, dtree_nodes[is_leaves[dtree_nodes]])


def get_sample_ids_with_unique_subpatterns(dtree, samples_pc):
    leaf_ids = dtree.apply(samples_pc)
    uleaves = np.unique(leaf_ids)

    upattern_sample_inds = []
    for uleaf in uleaves:
        sample_ind = np.argmax(leaf_ids == uleaf)
        upattern_sample_inds.append(sample_ind)

    # ensure that that the sample indices are in ascending order
    upattern_sample_inds.sort()

    return upattern_sample_inds


def get_neurons_in_subpatterns(dtree, unique_pattern_samples):
    '''
    returns list of subpattern neurons
    '''
    feature = dtree.tree_.feature
    _, _, are_leaves = get_tree_structure(dtree)

    node_indicator = dtree.decision_path(unique_pattern_samples)

    subpattern_neurons = []
    for i in range(len(unique_pattern_samples)):
        # sample i on row i has visited dtree nodes in columns indices[indptr[i:i+1]]
        # see https://i.stack.imgur.com/12bPL.png for clarification
        nodes_on_path = node_indicator.indices[node_indicator.indptr[i]: node_indicator.indptr[i+1]]

        # consider only nodes that are not leaves
        # this is probably always the last node since we are talking about trees, but just to be sure
        nodes_on_path = remove_leaf_nodes(nodes_on_path, are_leaves)
        neurons_on_path = feature[nodes_on_path]

        subpattern_neurons.append(neurons_on_path)

    return np.array(subpattern_neurons)


def get_unique_subpatterns_for_class(dtree, lps, targets, c: int):
    # the samples that satisfy the postcondition
    samples_pc = get_samples_wrt_postcondition(lps, targets, c)

    if np.sum(samples_pc) == -1:  # no samples for this class
        return np.array([[[], []]])

    # samples leading to unique leaves
    upattern_sample_inds = get_sample_ids_with_unique_subpatterns(
        dtree, samples_pc)
    assert (samples_pc[upattern_sample_inds[-1]] == samples_pc[upattern_sample_inds][-1]).all(
    ), 'it shouldnt matter if take a sample in index i or if we take samples resulting in unique subpattern and take the sample i from there '

    # samples that had unique pattern
    upattern_samples = samples_pc[upattern_sample_inds]

    # which neurons were considered with these samples
    subpattern_neurons = get_neurons_in_subpatterns(dtree, upattern_samples)

    subpatterns = []
    for i, lp in enumerate(upattern_samples):
        neuron_indx = subpattern_neurons[i]
        neuron_activations = lp[neuron_indx]
        subpatterns.append((neuron_indx, neuron_activations))

    return np.array(subpatterns)
