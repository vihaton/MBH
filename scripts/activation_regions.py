from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt

from .models import extract_pruning_mask


def get_random_classes(n_draw, n_total):
    # choose 3 random classes to draw the images from
    return np.random.choice(n_total, n_draw, replace=False)


def extract_bias_and_weights(module):
    '''returns two of the parameters, one of which has "weight" and another with "bias" in their name.'''
    biases, weights = None, None
    for pname, param in module.named_parameters():
        if 'bias' in pname:
            biases = param
        elif 'weight' in pname:
            weights = param
        else:
            print('We received a paramater that wasnt a weight or a bias!')
    return weights, biases


class hooking_1D_AR():
    def __init__(self, module, is_input_layer=False, is_output_layer=False):
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        self.is_input_layer = is_input_layer
        self.is_output_layer = is_output_layer
        self.flush_regions()

    def flush_regions(self):
        self.regions = []
        # there is always at least one region [origin, input] <--- this is added later in counting

    def hook_fn(self, module, input):  # module is a layer
        # let's unwrap input
        if isinstance(input, tuple):
            input = input[0]

        with torch.no_grad():
            weights, biases = extract_bias_and_weights(module)

            buffers = list(module.buffers())
            if len(buffers) == 0:
                buffer = None  # this hasn't been pruned yet
            elif len(buffers) == 1:
                buffer = buffers[0]
            else:
                print('We have more buffers than expected. Expected 1 buffer, got {}'.format(
                    len(buffers)))

            for n_node in range(len(weights)):
                if not self.is_output_layer and buffer is not None and (buffer[n_node].sum() == 0).item():
                    continue

                #print(f'\tnode {n_node}')
                w, b = weights[n_node], biases[n_node]

                # Where does the line z(x) == b cut the input vector?
                #   Lets solve the multiplier 'reg_cut = r', for which r * input is the point where
                #   the line z(x) == b crosses the infinity long vector that is parallel to the input vector.
                #   If r e [0,1], then the input vector is cut from its length
                #   -> there is a new activation region in that subspace

                #    z(r * input) == b
                # <=> r * z(input) == b
                # <=>            r == b / z(input)

                z = (input * w).sum()  # pre-activation z(x)

                reg_cut = torch.div(b, z).item()

                if reg_cut > 0 and reg_cut < 1:
                    # it is sufficient to save only the multiplier to mark the point where the lines crossed
                    self.regions.append(reg_cut)
                #print(f'\t\tthe 1D line would be cut at {reg_cut}')
                #print(f'\t\tweight {w.shape} bias {b}')

    def close(self):
        self.hook.remove()


def close_hooks(hooks):
    for hook in hooks:
        hook.close()


def flush_regions_from_hooks(hooks):
    for hook in hooks:
        hook.flush_regions()


def hook_layers(net, hooking_object):
    hooks = []
    kids = list(net.children())
    n_layers = len(kids)
    for l, layer in enumerate(kids):
        is_input_layer = l == 0
        is_output_layer = l == n_layers - 1
        hook = hooking_object(layer, is_input_layer, is_output_layer)
        hooks.append(hook)
    return hooks


def compute_1D_ARs(net, data_loader, average_over=100):
    '''
    Goes through `average_over` first images of the `data_loader`, and collects
    the 1D activation region borders for each image.
    '''
    batch_size = data_loader.batch_size
    net.eval()

    # add hooks to layers
    hooks = hook_layers(net, hooking_1D_AR)

    regions = []
    i_image = 0
    for i, (data, target) in enumerate(data_loader):
        data = data.reshape(batch_size, -1)  # flatten
        for image in data:
            net(image)  # feed image in

            regs_for_sample = []
            for hook in hooks:  # collect the regions
                regs_for_sample.append(hook.regions.copy())
                hook.flush_regions()

            regions.append(regs_for_sample)

            i_image += 1
            if i_image == average_over:
                close_hooks(hooks)
                return regions

    # we will arrive here only if average_over > amount of images in total
    close_hooks(hooks)
    return regions


def count_regions(one_dim_regions):
    '''input [#images, #layers, n_regions]

    return for (whole network), [by layers]
        (ave, min, max), [#layers, (ave, min, max)]

    '''
    n_regs = []
    for sample in one_dim_regions:
        n_reg = []
        for reg in sample:
            n_reg.append(len(reg))
        n_regs.append(n_reg)

    n_regs = np.array(n_regs)

    n_regs_all = np.sum(n_regs, axis=1)  # sum the regions of the whole network

    def ave_min_max(n_regs):
        return np.array((
            np.mean(n_regs, axis=0),
            np.min(n_regs, axis=0),
            np.max(n_regs, axis=0)
        )).T + 1  # there is always at least 1 region, which is then further cut by the AR borders

    return ave_min_max(n_regs_all), ave_min_max(n_regs)


def average_region_counts_over_models(models, data_loader, average_over_images=100):
    results_whole, results_layers = [], []
    for i, model in enumerate(models):
        regions_1d = compute_1D_ARs(
            model, data_loader, average_over=average_over_images)
        r_w, r_l = count_regions(regions_1d)
        results_whole.append(r_w)
        results_layers.append(r_l)

    return np.mean(results_whole, axis=0), np.mean(results_layers, axis=0)


def region_counts_1D(models, data_loader, average_over_images=100):
    '''
    models key:architecture, key:params
        -> #hyperparams, #n_models
    '''
    region_counts = OrderedDict()
    region_counts_layers = OrderedDict()

    for name in models:
        rcs_by_prs, rcsl_by_prs = [], []
        for params in models[name]:
            print(name, params)
            rcs_by_h, rcsl_by_h = [], []
            for models_n in models[name][params]:
                rcs, rcs_l = average_region_counts_over_models(models=models_n,
                                                               data_loader=data_loader,
                                                               average_over_images=average_over_images)
                rcs_by_h.append(rcs)
                rcsl_by_h.append(rcs_l)
            rcs_by_prs.append(np.array(rcs_by_h))
            rcsl_by_prs.append(np.array(rcsl_by_h))
        region_counts[name] = np.array(rcs_by_prs)
        region_counts_layers[name] = np.array(rcsl_by_prs)
    return region_counts, region_counts_layers


def region_counts_1D_for_models(models, data_loader, is_sparse, average_over_images=100):
    region_counts = OrderedDict()
    region_counts_layers = OrderedDict()

    for name in models:
        if not is_sparse:
            rcs, rcs_l = average_region_counts_over_models(models=models[name],
                                                           data_loader=data_loader,
                                                           average_over_images=average_over_images)
            region_counts[name] = rcs
            region_counts_layers[name] = rcs_l
        else:  # we have sparse models
            rcs_by_prs, rcsl_by_prs = [], []
            for ms in models[name]:
                rcs, rcs_l = average_region_counts_over_models(models=models[name][ms],
                                                               data_loader=data_loader,
                                                               average_over_images=average_over_images)
                rcs_by_prs.append(rcs)
                rcsl_by_prs.append(rcs_l)
            region_counts[name] = np.array(rcs_by_prs)
            region_counts_layers[name] = np.array(rcsl_by_prs)
    return region_counts, region_counts_layers


def compute_number_of_ars_from_regions_over_training(regions):
    '''
    input shape
        #models, #evaluations, average_over_images, #layers [AR data]
    '''
    n_models = regions.shape[0]
    ars, arls = [], []
    for i_m in range(n_models):
        ars_m, arls_m = [], []
        for i_e in range(regions.shape[1]):
            regs_e = regions[i_m, i_e]
            ar, arl = count_regions(regs_e)
            ars_m.append(ar)
            arls_m.append(arl)
        ars.append(ars_m)
        arls.append(arls_m)
    ars = np.array(ars)
    arls = np.array(arls)
    return ars, arls


## 2D ##

def project_to_2D_space(point, base):
    offset, xaxis, yaxis = base
    point_m = point - offset  # move to "origin"
    x = (point_m @ xaxis.T) / np.linalg.norm(xaxis)
    y = (point_m @ yaxis.T) / np.linalg.norm(yaxis)
    return np.array((x, y))


def project_to_higher_dim(point_2d, base):
    offset, xaxis, yaxis = base
    return offset + point_2d[0] * xaxis + point_2d[1] * yaxis


def project_points_to_higher_dim(points_2d, base):
    return np.array([project_to_higher_dim(point, base) for point in points_2d])


def test_interpolation(points_2d, points_input_space, n, base, verbose=False):
    n = round(n, 4)  # numerical instability is a challenge
    if verbose:
        print('n', n)
    assert n >= 0 and n <= 1, f'the split should be between these two points, it cannot be outside of [0,1], but n was {n}'
    interp_2d = points_2d[0] + n*(points_2d[1]-points_2d[0])
    interp_higher = points_input_space[0] + n * \
        (points_input_space[1]-points_input_space[0])
    interp_higher = project_to_2D_space(interp_higher, base)

    if verbose:
        print('interpolation 2d\t\t\t', interp_2d)
    if verbose:
        print('interp. in input space, then to 2d\t', interp_higher)

    diff = np.linalg.norm(interp_higher - interp_2d)
    if np.abs(diff) > 0.0001:
        print(
            f'##¤¤##¤¤##\tthe transform is not numerically stable, the difference between two equally valid methods was {diff}')

    if verbose:
        print('diff', diff)


def angle_between_two_vectors(v1, v2):
    # make unit vectors to avoid numerical errors
    len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if len1 != 0:
        v1 = v1 / len1
    if len2 != 0:
        v2 = v2 / len2
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def area_between_two_vectors(v1, v2):
    ang = angle_between_two_vectors(v1, v2)
    # divide by two bc we want the area between, not the rectangle spun by the two vectors
    return np.linalg.norm(v1) * np.linalg.norm(v2) * np.sin(ang) / 2


def calculate_area_of_region(reg):
    points = reg.points_high_dim
    first = points[0]

    area = 0
    for i in range(1, len(points)-1):
        p1, p2 = points[i], points[i+1]
        v1, v2 = p1 - first, p2 - first
        sector_area = area_between_two_vectors(v1, v2)
        area += sector_area

    return area


class Region():
    # TODO what to do with a (very theoretical and unlikely situation with real networks) case of
    # two neuron projections being exactly the same --> they will split the same regions (and increase the region count),
    # even though in the 2d space they don't create any new visible regions?

    def __init__(self, points, base=None, activations=None, verbose=False):
        # points is an array where there is a border between to consecutive points, also the first and the last.
        self.points = np.array(points)
        # layers, neurons
        self.activations = [[]] if activations is None else activations
        self.point_dim = self.points.shape[-1]
        self.verbose = verbose
        self.base = np.array([(0, 0), (1, 0), (0, 1)]
                             ) if base is None else base
        self.points_high_dim = project_points_to_higher_dim(
            self.points, self.base)
        self.area = None  # compute only if needed

        # this could be done more efficiently, if the transformations would be saved globally instead of per region TODO
        # for dynamic programming: save transformations deeper into the network
        self.transformed_points = []

        assert len(self.points) == len(self.points_high_dim)

    def reset(self):
        self.activations = [[]]
        self.transformed_points = []

    def get_area(self):
        if self.area is None:
            self.area = calculate_area_of_region(self)
        return self.area

    def add_neuron_activation(self, activation_status: int, layer: int):
        if len(self.activations) > layer:
            self.activations[layer].append(activation_status)
        else:
            self.activations.append([activation_status])

    def append_neuron_activation(self, activations, activation_status: int, layer: int):
        if len(activations) > layer:  # the layer is new
            activations[layer].append(activation_status)
        else:
            activations.append([activation_status])
        return activations

    def satisfies_subpattern(self, layer_i, neuron_indx, statuses):
        a = self.get_activations(layer_i)
        if max(neuron_indx) >= len(a):
            print(
                f'### We received a neuron index {max(neuron_indx)} that didnt exist for the layer {layer_i}!')
            return False

        return all(a[neuron_indx] == statuses)

    def remove_activations(self):
        self.activations = [[]]

    def get_activations(self, layer_i=None):
        '''
        Casting lists to numpy arrays here causes performance problems.
        Instead, we know that neurons are added layer by layer -> once we
        have neuron added on a deeper layer, we can cast earlier one to
        array once.

        layer_i == -1  --> last hidden layer
        otherwise layer_i should be [0, #hidden layers]
        '''

        if layer_i is not None:
            return self.activations[layer_i]
        else:
            return np.array(self.activations)

    def map_to_activation_status(self, sign: int):
        '''
        activation function: ReLU
        '''
        if self.verbose:
            assert sign == - \
                1 or sign == 1, f'sign should be -1 or 1, not {sign}'
        actstat = 0 if sign == -1 else 1
        return actstat

    def get_points_for_plotting(self):
        '''
        adds the first point to the end of the array as well
        '''
        return np.append(self.points, self.points[0]).reshape(-1, self.point_dim)

    def split(self, w, b, points_tra, signs):
        '''
        solve the exact intersections between the neuron (w,b) and the region defined by corner points
        that are transformed to the input space of the layer this neuron lies on.
        '''
        new_regions_signs = np.zeros(2)

        def append_intersection(splits, intersection, split_ind):
            # the new intersection point defines both splits
            splits[split_ind].append(intersection)
            # following points belong to another split
            split_ind = (split_ind + 1) % 2
            splits[split_ind].append(intersection)
            return split_ind

        splits = [[], []]
        split_ind = 0
        prev = -1
        for i, sign in enumerate(signs):
            # either 1) the previous sign was different so there is a cut between the two points...
            if np.abs(signs[prev] - sign) == 2:
                # this border was cut in between the corner points, lets solve the exact intersection

                p0, p1 = points_tra[i], points_tra[prev]

                if self.verbose and p0.shape[0] < 10:
                    print(f'\tr_i {self.points[i]} => {p0}')
                    print(
                        f'\tr_j {self.points[prev]} => {p1} transformed for this neuron w {w} b {b}')

                # there exists n, for which (p0 + n(p1 - p0)) @ w.T + b == 0
                # => n = - (p0 @ w.T + b) / (p1-p0) @ w.T
                # where the (p0 + n(p1-p0)) is the intersection point
                nom = - (p0 @ w.T + b)
                denom = (p1-p0) @ w.T
                n = nom / denom
                inters = p0 + n * (p1-p0)

                if self.verbose:
                    assert round(inters @ w.T + b, 3) == 0, f'neuron defines a hyperplane where it gets value 0,\
 and this intersection should be on that hyperplane! Instead got {inters @ w.T - b}'
                    print(f'intersection {inters}, n {n}')

                # because the transformation inside one region is linear, and all the previous layer neurons
                # have been evaluated (so we know this is region is atomic wrt. to the preceeding unlinearities)
                # we can interpolate the intersection using the `n` from above.
                p0, p1 = self.points[i], self.points[prev]
                inters_2d = p0 + n * (p1-p0)

                new_regions_signs[split_ind] = signs[prev]
                split_ind = append_intersection(splits, inters_2d, split_ind)
                new_regions_signs[split_ind] = sign

                if self.verbose:
                    print('\tintersection between ', p0, p1)
                    print('\tintersection 2d', inters_2d)

            # ...or 2) the intersection is on this point
            # AND the cut is not parallel to either of the lines leaving this point (i.e. it crosses the region).
            # The latter has been tested earlier, here we know that the new border actually crosses this region.
            if sign == 0:
                new_regions_signs[split_ind] = signs[prev]
                split_ind = append_intersection(
                    splits, self.points[i], split_ind)
                new_regions_signs[split_ind] = - signs[prev]

                if self.verbose:
                    print(f'\tintersection on 2d point {self.points[i]}')
            else:  # if there was no intersection on this point, i.e. sign != 0, then this point should be added to a region
                # this point belongs to the split denoted by the index
                splits[split_ind].append(self.points[i])

            prev = i

        if self.verbose:
            print('region was split to two new splits', splits)
        return np.array(splits), new_regions_signs

    def get_transformed_points(self, layer_i, prev_layers):
        # for each transformed layer we get one list of points
        n_trans = len(self.transformed_points)
        if layer_i - 1 < n_trans:  # if we computed these transforms earlier
            if self.verbose:
                print('used dynamic programming to fetch transformations of shape',
                      self.transformed_points[layer_i - 1].shape)
            return self.transformed_points[layer_i - 1]

        if self.verbose:
            print(
                f'we need transformed points for layer {layer_i}, the current ones', self.transformed_points)

        # take the most deep transformation we have or base case the input space one
        points_transf = self.transformed_points[-1] if n_trans > 0 else self.points_high_dim
        if self.verbose:
            print('previously transformed points shape', points_transf.shape)

        # change to tensor, apply non-linearity
        points_t = torch.tensor(np.float32(points_transf))
        for i in range(n_trans, layer_i):
            # we have transformations until the layer i, now we need i+1
            layer = prev_layers[i]
            if self.verbose and points_t.shape[0] < 10:
                print(
                    f'apply layer {i}, layer {layer}, points that need to be transformed {points_t}')
            preact = layer(points_t)
            points_t = torch.nn.functional.relu(preact)
            self.transformed_points.append(points_t.detach().numpy())

            if self.verbose and points_t.shape[0] < 10:
                print(f'transformed points {points_t}')

        if self.verbose:
            print('points transformed')
        return points_t.detach().numpy()

    def cast_layer_activations_to_numpy(self, layer_i):
        '''layer_i [0,n]'''
        self.activations[layer_i] = np.array(self.activations[layer_i])
        if self.verbose:
            print('cast previous layer activations to numpy array, there will be no more additions to the prev layer',
                  self.activations[layer_i])

    def get_corner_point_signs(self, res):
        signs = np.sign(res)
        u_signs = list(set(signs))
        if self.verbose:
            print('signs', signs)

        if len(u_signs) == 1 and u_signs[0] == 0:
            if self.verbose:
                print(
                    'the whole region has collapsed to one point in the transformed space of some later layer(s)')
            u_signs = np.array([-1])  # we can put it manually off

        if self.verbose:
            print('unique signs', u_signs)

        return signs, u_signs

    def update_with_neuron(self, w, b, prev_layers):
        '''
        The weight vector should be in the original dim. Previous layers provide the transformations needed to split these regions.

        returns if the region was split + the resulting region(s):
            True, the two new regions produced by the split;
            False, the current region
        '''
        layer_i = len(prev_layers)
        if layer_i == len(self.activations) and type(self.activations[layer_i-1]) is list:
            self.cast_layer_activations_to_numpy(layer_i-1)

        if self.verbose and w.shape[0] < 10:
            print(
                f'\nupdate region {self.points} with w {w} b {b} from layer {layer_i}')

        if layer_i > 0:
            # transfer the points to the input space of this neuron
            points = self.get_transformed_points(layer_i, prev_layers)
        else:
            points = self.points_high_dim

        if self.verbose and w.shape[0] < 10:
            print(
                f'points transferred to the input space of this layer {points}')

        if self.verbose:
            assert w.shape[0] == points.shape[
                1], f'weight vector should be the same dim as the point vector that defines the region, they were {w.shape} {points.shape}'

        res = points @ w.T + b
        if self.verbose:
            print('res', res)

        signs, u_signs = self.get_corner_point_signs(res)

        splits_the_region = -1 in u_signs and 1 in u_signs

        if not splits_the_region:
            # since the region is not split, there is either 1 or 2 unique signs (with and w/o zero)
            sign = u_signs[0]
            if sign == 0:
                sign = u_signs[-1]
            self.add_neuron_activation(
                self.map_to_activation_status(sign), layer_i)
            return False, self
        else:
            new_regions_points, new_regions_signs = self.split(w, b,
                                                               points_tra=points,
                                                               signs=signs)
            act_statuses = map(
                self.map_to_activation_status, new_regions_signs)
            if self.verbose:
                print('activation statuses', act_statuses)
            acts_a, acts_b = [
                self.append_neuron_activation(deepcopy(self.activations),
                                              ast,
                                              layer=layer_i) for ast in act_statuses
            ]
            ra = Region(
                new_regions_points[0], base=self.base, activations=acts_a, verbose=self.verbose)
            rb = Region(
                new_regions_points[1], base=self.base, activations=acts_b, verbose=self.verbose)
            return True, (ra, rb)

    def __str__(self):
        return f'{self.points}\n\tactivations {self.activations}\n'


class Subspace():
    def __init__(self, base_region, base, spanning_images):
        self.regions = [base_region]
        self.base_region = base_region
        self.base = base
        self.spanning_images = spanning_images
        self.spanning_images_2D = project_examples_into_2d(
            spanning_images, base)
        self.n_neurons = 0
        self.local_area = calculate_area_of_region(base_region)

    def remove_splits(self):
        self.base_region.reset()
        self.regions = [self.base_region]
        self.n_neurons = 0

    def number_of_regions(self):
        # TODO make sure that regions with exactly same corners are not counted twice
        return len(self.regions)

    def get_mask_that_satisfies_subpattern(self, layer_i, neuron_indx, statuses):
        if len(neuron_indx) == 0:  # there is no requirements
            return np.ones(len(self.regions))
        return np.array([reg.satisfies_subpattern(layer_i, neuron_indx, statuses) for reg in self.regions])

    def get_areas_of_regions(self, mask):
        assert len(mask) == len(self.regions), 'the mask is wrong size'
        return np.array([
            self.regions[i].get_area() for i in range(len(self.regions)) if mask[i]
        ])

    def local_area_of_regions_that_satisfy_subpattern(self, layer_i, neuron_indx, statuses):
        satisfy_mask = self.get_mask_that_satisfies_subpattern(
            layer_i, neuron_indx, statuses)
        areas = self.get_areas_of_regions(satisfy_mask)
        return np.sum(areas)

    def split_by_neuron(self, weights, bias, pruning_mask, prev_layers):
        # apply the sparsity
        weights[pruning_mask == 0] = 0

        new_regions = []
        for reg in self.regions:
            did_split, regs = reg.update_with_neuron(
                weights, bias, prev_layers)
            if did_split:
                new_regions.append(regs[0])
                new_regions.append(regs[1])
            else:
                new_regions.append(regs)
        self.regions = new_regions
        self.n_neurons = self.n_neurons + 1

    def cast_layer_activations_to_numpy(self, layer_i=-1):
        '''
        layer_i == -1 stands for the last HIDDEN layer
        '''
        for reg in self.regions:
            reg.cast_layer_activations_to_numpy(layer_i)

    def split_by_model(self, model, split_with_output_layer=False):
        layers = list(model.children())
        if not split_with_output_layer:
            layers = layers[:-1]        # don't run the output layer!

        prev_layers = []
        for i, layer in enumerate(layers):
            with torch.no_grad():
                weights, biases = extract_bias_and_weights(layer)
                pruning_mask = extract_pruning_mask(layer)
                if pruning_mask is None:
                    pruning_mask = torch.ones_like(weights)

                for n_node in range(len(weights)):
                    w, b, pm = weights[n_node].detach().numpy(), biases[n_node].detach(
                    ).numpy(), pruning_mask[n_node].detach().numpy()
                    self.split_by_neuron(w, b, pm, prev_layers)

            prev_layers.append(layer)

        # after applying the last layer, cast all the activations on the las hidden layer to numpy arrays
        # while doing this here once, we dont need to do it somewhere else several times
        self.cast_layer_activations_to_numpy()

        return self

    def __str__(self):
        return f'base region\n{self.base_region} #regions {self.number_of_regions()}'

    def visualize(self, pois=[], title=None, colors=None, cmap=None, draw_spanning_imgs=True, fill=True, markers=None,
                  edgecolor=None, sample_n_regs=None, linewidth=0):
        if len(pois) == 0 and draw_spanning_imgs:
            pois = self.spanning_images_2D
        if title is None:
            title = f'{self.number_of_regions()} local regions defined by {self.n_neurons} neurons'

        if sample_n_regs is not None and sample_n_regs < self.number_of_regions():
            rnd_inds = np.random.randint(
                self.number_of_regions(), size=sample_n_regs)
            regs = np.array(self.regions)[rnd_inds]
            colors = colors[rnd_inds]
        else:
            regs = self.regions

        draw_regions(regs, pois=pois, title=title,
                     colors=colors, cmap=cmap, fill=fill, markers=markers,
                     edgecolor=edgecolor, linewidth=linewidth)


def visualize_class_blankets_several_coverages(subspace, coverages, max_patterns, suptitle=None,
                                               points=None, point_colors=None, point_classes=None,
                                               filename=None, save_figures=False,
                                               figsize=None, dpi=None,
                                               ylim=None, xlim=None,
                                               class_names=None, hyperp=None,
                                               sample_n_regs=None):

    n_n = rows = len(coverages)
    cols = len(max_patterns[0])  # classes

    figsize = figsize if figsize is not None else (4.2*cols, 4*rows)
    dpi = dpi if dpi else 100
    fig = plt.figure(figsize=figsize, dpi=dpi)

    subs_area = subspace.local_area

    for i, coverage in enumerate(coverages):
        max_pattern_cov = max_patterns[i]
        print(f"coverage {coverage}, {i+1}/{len(coverages)}")

        for j, maxp_class in enumerate(max_pattern_cov):
            print(f'class {j+1}', end='\r')
            plt.subplot(rows, cols, i*cols + j + 1)

            n_indx, n_status = maxp_class
            satisfy_mask = subspace.get_mask_that_satisfies_subpattern(
                -1, n_indx, n_status)
            class_cover_area = subspace.local_area_of_regions_that_satisfy_subpattern(
                -1, n_indx, n_status)

            title = f'blanket covers {class_cover_area/subs_area*100:.0f}%, #constraints={len(n_indx)}'

            subspace.visualize(title=title,
                               colors=satisfy_mask*1,
                               draw_spanning_imgs=True,
                               markers=markers[:3],
                               sample_n_regs=sample_n_regs,
                               linewidth=.2)

            if points is not None:
                cla = j-1 if point_classes is not None and j not in point_classes else j
                ps = points if point_classes is None else points[point_classes == cla]

                if point_colors is not None:
                    c = point_colors if point_classes is None else point_colors[point_classes == cla]
                    plt.scatter(ps[:, 0], ps[:, 1], s=5,
                                alpha=0.5, c=c, zorder=2),
                else:
                    plt.scatter(ps[:, 0], ps[:, 1], s=5, alpha=0.5, zorder=2),

            ax = plt.gca()

            if i == rows - 1:
                xlabel = f'class {j}' if class_names is None else f'class {j} {class_names[j]}'
                ax.set(xlabel=xlabel)

            if j == 0:
                eqstr = '>' if coverage < 1 else ''
                ax.set(ylabel=f'class coverage {eqstr}= {coverage*100:.0f}%')

            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.tick_params(axis='both', which='both', length=0)

            if ylim is not None and xlim is not None:
                plt.ylim(ylim)
                plt.xlim(xlim)
            else:
                plt.axis('equal')

    suptitle = f'Class blankets with coverages {coverages} for a network with {hyperp[0]} neurons' if suptitle is None else suptitle
    plt.suptitle(suptitle, y=1.01)
    plt.tight_layout()
    fig.patch.set_facecolor('w')
    if filename and save_figures:
        plt.savefig(filename)
        plt.show()
    else:
        plt.show()


def visualize_one_class_blanket(subspace, coverages, max_patterns, cov_i, class_i,
                                classes=None,
                                suptitle=None,
                                points=None, point_colors=None, point_classes=None,
                                filename=None, save_figures=False,
                                figsize=None, dpi=None,
                                ylim=None, xlim=None,
                                class_names=None, hyperp=None):

    figsize = figsize if figsize is not None else (7, 7)
    dpi = dpi if dpi else 160
    fig = plt.figure(figsize=figsize, dpi=dpi)

    markers = ['o', 's', '^']
    coverage = coverages[cov_i]

    subs_area = subspace.local_area

    n_indx, n_status = max_patterns[cov_i][class_i]
    satisfy_mask = subspace.get_mask_that_satisfies_subpattern(
        -1, n_indx, n_status)
    class_cover_area = subspace.local_area_of_regions_that_satisfy_subpattern(
        -1, n_indx, n_status)

    title = f'blanket covers {class_cover_area/subs_area*100:.0f}% of the 2D plane'

    if False and classes is not None and class_i in classes:
        classes = np.array(classes) if type(
            classes) is not np.ndarray else classes
        c_ind = (classes == class_i).argmax()
        pois = subspace.spanning_images_2D[c_ind][None]
        m = markers[c_ind]
    else:  # always use the three, it's more clear that way
        pois = subspace.spanning_images_2D
        m = markers

    subspace.visualize(title=title,
                       pois=pois,
                       colors=satisfy_mask*1,
                       draw_spanning_imgs=True,
                       markers=m,
                       edgecolor='k',
                       linewidth=.1)

    if points is not None:
        cla = j-1 if point_classes is not None and j not in point_classes else j
        ps = points if point_classes is None else points[point_classes == cla]

        if point_colors is not None:
            c = point_colors if point_classes is None else point_colors[point_classes == cla]
            plt.scatter(ps[:, 0], ps[:, 1], s=5, alpha=0.5, c=c, zorder=2),
        else:
            plt.scatter(ps[:, 0], ps[:, 1], s=5, alpha=0.5, zorder=2),

    xlabel = f'class {class_i}' if class_names is None else f'class {class_i} {class_names[class_i]}'
    plt.xlabel(xlabel)

    eqstr = '>' if coverage < 1 else ''
    plt.ylabel(f'class coverage {eqstr}= {coverage*100:.0f}%')

    ax = plt.gca()

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.tick_params(axis='both', which='both', length=0)

    if ylim is not None and xlim is not None:
        plt.ylim(ylim)
        plt.xlim(xlim)
    else:
        plt.axis('off')

    suptitle = f'Network with {hyperp[0]} neurons, #constraints={len(n_indx)}' if suptitle is None else suptitle
    plt.suptitle(suptitle, y=0.999)
    plt.tight_layout()
    fig.patch.set_facecolor('w')
    if filename and save_figures:
        plt.savefig(filename)
        plt.show()
    else:
        plt.show()


def visualize_class_blankets(subspace, model, hyperp, max_patterns, lp_samples, class_coverage, filename=None, save_figures=False):
    fig = plt.figure(figsize=(12, 6*5))

    tot_area = subspace.local_area
    l_neur = hyperp[0][-2]  # last hidden layer is layer -2

    for c, mpattern in enumerate(max_patterns):
        plt.subplot(5, 2, c+1)
        n_indx, n_status = mpattern
        satisfy_mask = subspace.get_mask_that_satisfies_subpattern(
            -1, n_indx, n_status)
        class_cover_area = subspace.local_area_of_regions_that_satisfy_subpattern(
            -1, n_indx, n_status)

        subspace.visualize(title=f'Subpattern ({len(mpattern[0])}/{l_neur} neurons) that covers {class_coverage*100:.0f}% of\n{lp_samples} class {c} samples. Local area {class_cover_area:.1f} ({class_cover_area/tot_area*100:.2f}%)',
                           colors=satisfy_mask*1)
        plt.axis('equal')

    plt.suptitle(
        f'Activation of subpattern on the last hidden layer. Coverage {class_coverage}. Subspace area {tot_area:.1f}', y=1)
    plt.tight_layout()
    fig.patch.set_facecolor('w')

    if filename and save_figures:
        plt.savefig(filename)
    else:
        plt.show()


def compute_and_visualize_local_2D_ARs(subspace, models, sparse_from, param_keys, n_models, hyperparams_for_sparse,
                                       axis_off=False, filename=None, save_figures=False):
    for j, params in enumerate(param_keys[sparse_from]):
        n_m = n_models[sparse_from]
        hyperp = hyperparams_for_sparse[sparse_from][j]
        n_h = len(hyperp)

        fig = plt.figure(figsize=(n_m*6, n_h*6))

        for i, models_n in enumerate(models[sparse_from][params]):
            for n, model in enumerate(models_n):
                subspace.remove_splits()
                plt.subplot(n_h, n_m, i*n_m + n + 1)

                subspace = subspace.split_by_model(model)
                subspace.visualize()
                plt.axis('equal')
        plt.suptitle(
            f'local 2D ARs by models with {params/1000:.1f}k parameters', y=1)
        fig.patch.set_facecolor('w')
        plt.tight_layout()

        if filename and save_figures:
            plt.savefig(filename.format(j))
        else:
            plt.show()


def compute_local_2D_ARs(model, use_three_samples, data_loader, classes, average_over_images):
    stats = []
    n_classes = len(data_loader.dataset.classes)
    random_classes = classes is None

    for i in range(average_over_images):
        if random_classes:
            classes = get_random_classes(3, n_classes)

        example_imgs = get_example_images(data_loader, classes)
        example_imgs = example_imgs.reshape(3, -1)
        assert len(
            example_imgs.shape) == 2, f'example imgs shape {example_imgs.shape}'
        subspace = span_subspace_with_three_images(
            example_imgs, use_three_samples=use_three_samples)

        plane_area = subspace.local_area

        subspace = subspace.split_by_model(model)
        n_r = subspace.number_of_regions()

        # TODO is there other metrics we would like to compute?
        stats.append((n_r / plane_area, n_r, plane_area))

    stats = np.array(stats)
    return np.array((np.mean(stats, axis=0), np.min(stats, axis=0), np.max(stats, axis=0))).T


def compute_local_2D_ARs_for_models(models, use_three_samples, data_loader, classes,
                                    average_over_images, param_keys,
                                    n_models, hyperparams_for_sparse,
                                    models_total, verbose=False):
    m_i = 0
    ar_stats = OrderedDict()
    for name in models:
        ars_param = []
        for j, params in enumerate(param_keys[name]):
            ars_n = []
            for i, models_n in enumerate(models[name][params]):
                ars = []
                for n, model in enumerate(models_n):
                    m_i += 1
                    if verbose:
                        print(
                            f'computing stats for model {m_i}/{models_total}', end='\r')
                    stats_m = compute_local_2D_ARs(model, use_three_samples,
                                                   data_loader, classes,
                                                   average_over_images)
                    ars.append(stats_m)

                ars_n.append(ars)
            ars_param.append(np.array(ars_n))
        ar_stats[name] = np.array(ars_param)

    return ar_stats


def get_projection_base(example_imgs, use_three_samples=True):
    assert len(example_imgs) == 3
    if len(example_imgs.shape) > 2:  # flatten
        example_imgs = example_imgs.reshape(3, -1)

    axisx = example_imgs[1].copy()
    axisy = example_imgs[2].copy()
    offset = np.zeros_like(axisx)

    if use_three_samples:
        offset = example_imgs[0].copy()
        axisx -= offset
        axisy -= offset

    yx = axisy @ axisx.T
    if yx != 0:
        # set y to be orthogonal to x
        axisy -= (axisy @ axisx.T) / (axisx @ axisx.T) * axisx
        assert axisy.shape == axisx.shape, f'after orthogonalizing axisy shape {axisy.shape} was supposed to be {axisx.shape}'

    # normalize
    axisx /= np.linalg.norm(axisx)
    axisy /= np.linalg.norm(axisy)

    assert round(np.linalg.norm(
        axisx), 3) == 1, f'length of x should be 1, was {np.linalg.norm(axisx)}'
    assert round(np.linalg.norm(
        axisy), 3) == 1, f'length of y should be 1, was {np.linalg.norm(axisy)}'
    assert round((axisx @ axisy), 3) == 0, 'x and y should be orthogonal'

    base = np.array((offset, axisx, axisy))
    spanning_images = np.array((offset, example_imgs[1], example_imgs[2]))
    return base, spanning_images


def project_examples_into_2d(example_imgs, base):
    return np.array([project_to_2D_space(img, base) for img in example_imgs])


def get_cornerns_of_a_subspace(examples_in_2d):
    centroid = np.mean(examples_in_2d, axis=0)
    diff = examples_in_2d-centroid
    dist_max = 1.2 * np.array(np.max(np.abs(diff)))
    diff = np.repeat(dist_max, 2)

    square_corners = [(centroid + np.array(((-1)**(n-2 < 0),
                                            (-1)**(n % 3 == 0))) * diff) for n in range(4)]

    return square_corners, centroid


def span_subspace_with_three_images(example_imgs, use_three_samples=True, verbose=False):
    base, spanning_images = get_projection_base(
        example_imgs, use_three_samples)

    spanning_in_2d = project_examples_into_2d(spanning_images, base)

    square_corners, _ = get_cornerns_of_a_subspace(spanning_in_2d)
    base_region = Region(square_corners, base=base, verbose=verbose)
    subspace = Subspace(base_region, base, spanning_images)

    return subspace


def get_example_images(data_loader, classes):
    # TODO should we try to get images from different parts of the space?
    for batch, targets in data_loader:
        indices = [np.argmax(targets == c).item() for c in classes]
        example_imgs = [batch[ind].detach().numpy() for ind in indices]
        break
    return np.array(example_imgs)


def visualize_example_images(example_imgs, classes, classes_str=None, horizontal=True):
    r, c = int(len(example_imgs) / 3), 3
    r, c = (r, c) if horizontal else (c, r)
    plt.figure(figsize=(5*c, 5*r))

    for j, img in enumerate(example_imgs):
        plt.subplot(r, c, j+1)
        plt.imshow(img.reshape(28, 28),
                   cmap='gray', vmin=0, vmax=1)
        if classes is not None:
            title = f'class {classes[j]} example image' if classes_str is None else f'class {classes_str[classes[j]]} example image'
            plt.title(title)
        plt.axis('off')
        plt.tight_layout()


def get_key_string(trained: bool, use_three_imgs: bool):
    training_str = 'trained' if trained else 'init'
    origin_str = 'orig' if not use_three_imgs else '3imgs'
    return f'{origin_str}_{training_str}'


def draw_regions(regions, pois=[], title='', color_by_activation_of_neuron_and_layer=None,
                 fill=True, colors=None,
                 xlim=None, ylim=None,
                 cmap=None, color_inds_cmap=None,
                 markers=None, edgecolor=None, linewidth=0):
    c_dict = {  # for coloring regions based on activation status of some neuron
        0: 'lightskyblue',
        1: 'r',
        2: 'y'
    }

    default_cmap_name = 'tab20'
    if cmap:
        sequencer = 10
    else:
        cmap = plt.get_cmap(default_cmap_name)
        sequencer = 20

    kwargs = {
        'alpha': 1,
        'linewidth': linewidth
    }
    if fill:
        draw_method = plt.fill
        if edgecolor is not None:
            kwargs['edgecolor'] = edgecolor
            kwargs['linewidth'] = 0
    else:
        draw_method = plt.plot

    list_color = False
    if colors is not None:
        assert len(colors) == len(
            regions), 'each region should have its color defined'
        list_color = True
    elif color_by_activation_of_neuron_and_layer is not None:
        ind, layer_i = color_by_activation_of_neuron_and_layer

    for i, r in enumerate(regions):
        arc = r.get_points_for_plotting()
        if list_color:
            # defined color from cdict
            c = c_dict[colors[i]]

            # use cmap
            #c = cmap(colors[i] % sequencer)
            draw_method(arc[:, 0], arc[:, 1], color=c, **kwargs)
        elif cmap:
            ind = color_inds_cmap[i] if color_inds_cmap else i
            draw_method(arc[:, 0], arc[:, 1], color=cmap(
                ind % sequencer), **kwargs)
        else:
            draw_method(arc[:, 0], arc[:, 1], c=c_dict[r.get_activations()[
                        layer_i][ind]], **kwargs)

        if fill and linewidth > 0:
            c = edgecolor if edgecolor is not None else 'k'
            plt.plot(arc[:, 0], arc[:, 1], linewidth=linewidth, c=c, zorder=2)

    # Points of interest
    if markers is None:
        markers = ['o']*len(pois)

    assert len(markers) == len(
        pois), 'number of points of interest and markers dont match'

    for marker, point in zip(markers, pois):
        plt.scatter(point[0], point[1], s=42,
                    marker=marker, color='k', edgecolors='w', zorder=3)

    plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
