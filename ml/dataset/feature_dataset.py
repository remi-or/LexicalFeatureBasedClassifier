# region Imports
from __future__ import annotations
from math import nan
from typing import List, Optional, Tuple
from itertools import chain

from .base import Dataset

import torch
from torch import Tensor
import numpy as np


# endregion

# region Feature class
class FeatureBatch:

    def __init__(
        self,
        X : Tensor,
        Y : Tensor,
    ) -> None:
        self.X = X
        self.Y = Y

    def to(
        self,
        device : str,
    ) -> None:
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)


class FeatureDataset(Dataset):

    def __init__(
        self,
        names : List[str],
        entries : List[List[float]],
        labels : List[int],
    ) -> None:
        self.names = names
        self.entries = entries
        self.labels = [int(label) for label in labels]
        self.yield_order = [i for i in range(len(self))]

    def deep_shuffle(
        self,
        seed : Optional[int],
    ) -> None:
        permutation = [i for i in range(len(self))]
        np.random.default_rng(seed).shuffle(permutation)
        self.entries = [self.entries[i] for i in permutation]
        self.labels = [self.labels[i] for i in permutation]

    def add(
        self,
        entries : List[List[float]],
        labels : List[int],
    ) -> None:
        assert len(self.entries[0]) == len(entries[0]), f"The entries present are of size {len(self.entries[0])} whereas the entries added are of size {len(entries[0])}."
        self.entries = self.entries + entries
        self.labels = self.labels + labels
        self.yield_order = [i for i in range(len(self))]
    
    def split(
        self,
        p : float,
        seed : Optional[int] = None,
    ) -> Tuple[FeatureDataset, FeatureDataset]:
        # Shuffling
        self.deep_shuffle(seed)
        # Preparing arguments
        d1 = {
            'names' : self.names,
            'entries' : self.entries[:int(p * len(self))],
            'labels' : self.labels[:int(p * len(self))],
        }
        d2 = {
            'names' : self.names,
            'entries' : self.entries[int(p * len(self)):],
            'labels' : self.labels[int(p * len(self)):],
        }
        # Create and return
        return FeatureDataset(**d1), FeatureDataset(**d2)

    def __len__(self) -> int:
        return len(self.labels)

    def number_of_features(self) -> int:
        return len(self.names)

    def number_of_labels(self) -> int:
        return 1 + max(self.labels)

    def proportions(self) -> Tensor:
        total = len(self.labels)
        props = [
            sum(1 for label in self.labels if label == target_label) / total
            for target_label in range(1 + max(self.labels))
        ]
        return torch.tensor(props)

    def balance(
        self,
        seed : Optional[int],
    ) -> Tuple[ List[List[float]], List[int] ]:
        self.deep_shuffle(seed)
        print(f"Length before balancing: {len(self)}")
        n = self.number_of_labels()
        labeled_indices = [[] for i in range(n)]
        for i in range(len(self)):
            labeled_indices[self.labels[i]].append(i)
        minority, new_indices, dropped = min(len(indexes) for indexes in labeled_indices), [], []
        for index in labeled_indices:
            for i in index[:minority]:
                new_indices.append(i)
            for i in index[minority:]:
                dropped.append(i)
        dropped = (
            [self.entries[i] for i in dropped],
            [self.labels[i] for i in dropped]
        )
        self.entries = [self.entries[i] for i in new_indices]
        self.labels = [self.labels[i] for i in new_indices]
        self.yield_order = [i for i in range(len(self))]
        print(f"Length after balancing: {len(self)}")
        return dropped

    def replace_na(
        self,
        replacement : float,
    ) -> None:
        for entry in self.entries:
            for i, value in enumerate(entry):
                if np.isnan(value):
                    entry[i] = replacement

    def normalize(
        self,
        ignore_na : bool = True,
    ) -> None:
        for j in range(self.number_of_features()):
            feature_min = min(self.entries[i][j] for i in range(len(self)))
            feature_max = max(self.entries[i][j] for i in range(len(self)))
            normalization_function = lambda x : (x - feature_min) / feature_max if feature_max != 0 else 1
            for i in range(len(self)):
                if not (ignore_na and np.isnan(self.entries[i][j])):
                    self.entries[i][j] = normalization_function(self.entries[i][j])
        
    def batches(
        self,
        batch_size : int = 1,
        shuffle : bool = True,
        seed : Optional[int] = None,
    ) -> FeatureBatch:
        if shuffle:
            self.shuffle(seed)
        for i in range(0, len(self), batch_size):
            X = torch.tensor([self.entries[j] for j in self.yield_order[i : i + batch_size]])
            Y = torch.tensor([self.labels[j] for j in self.yield_order[i : i + batch_size]])
            yield FeatureBatch(X, Y)

    