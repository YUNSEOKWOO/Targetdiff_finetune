import torch
from torch.utils.data import Subset
from .pl_pair_dataset import PocketLigandPairDataset
from .pdbbind import PDBBindDataset
from .pl_pair_dataset_finetune import PocketLigandPairDataset_finetune


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    elif name == 'pdbbind':
        dataset = PDBBindDataset(root, *args, **kwargs)
    elif name == 'finetune':
        dataset = PocketLigandPairDataset_finetune(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
