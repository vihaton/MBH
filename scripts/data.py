import torch
from torchvision import datasets, transforms
import numpy as np

# ## MNIST data


def get_dataset(data_name: str):
    if data_name == 'digit':
        return datasets.MNIST
    elif data_name == 'fashion':
        return datasets.FashionMNIST
    else:
        raise ValueError(f'unknown dataset name {data_name}')


def load_data(data_dir, data_name: str, batch_size, test_batch_size, mean=None, std=None, shuffle=True, kwargs={}):
    def normalize_features(image):
        if mean is None and std is None:
            return image

        out = image.view(-1, 28*28)
        # print('before norm', out.mean(), out.std(), out.min(), out.max())

        if mean is not None:
            out = (out - mean)

        if std is not None:
            out = out / std

        # print('after norm', out.mean(), out.std(), out.min(), out.max())
        return out.float()

    dataset = get_dataset(data_name)

    train_loader = torch.utils.data.DataLoader(
        dataset(data_dir, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(normalize_features),
                ])),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        dataset(data_dir, train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(normalize_features),
                ])),
        batch_size=test_batch_size, shuffle=shuffle, **kwargs)

    return train_loader, test_loader


def get_train_and_test_loaders(data_dir, data_name, batch_size, test_batch_size,
                               normalize_data, normalize_by_features, normalize_by_moving, normalize_by_scaling,
                               kwargs, device=None):
    train_loader, test_loader = load_data(
        data_dir, data_name, batch_size, test_batch_size, kwargs=kwargs)

    train_samples = train_loader.dataset.data.shape[0]
    if normalize_data:
        print('normalize the data')
        flat_train = train_loader.dataset.data.detach().view(train_samples, -1).numpy()

        if normalize_by_features:
            mean = np.mean(flat_train, axis=0) / 255
            std = np.std(flat_train, axis=0) / 255
            std[std == 0] = 1  # if the std is 0, we should divide with 1 instead
            # every feature has its own mean and std
            assert(len(mean) == len(std) == 784)
        else:  # use the same transformation to all of the features
            mean = np.mean(flat_train) / 255
            std = np.std(flat_train) / 255
            assert type(mean) == type(
                std) == np.float64, "there should be only one value for the whole dataset"

        print(type(mean), type(std))
        if normalize_by_scaling:
            print('std: shape, mean', std.shape, np.mean(std))
        else:
            std = None

        if normalize_by_moving:
            print('mean: shape, mean, max, min', mean.shape,
                  np.mean(mean), np.max(mean), np.min(mean))
        else:
            mean = None

        # mean = (0.1307,) # digit mnist
        # std = (0.3081,) # digit mnist

        train_loader, test_loader = load_data(data_dir,
                                              data_name, batch_size, test_batch_size, mean=mean, std=std, kwargs=kwargs)
    else:
        print('dont normalize')

    if device is not None:
        print('move data to device', device)
        train_loader.dataset.data = train_loader.dataset.data.to(device)
        train_loader.dataset.targets = train_loader.dataset.targets.to(device)

        test_loader.dataset.data = test_loader.dataset.data.to(device)
        test_loader.dataset.targets = test_loader.dataset.targets.to(device)

    return train_loader, test_loader
