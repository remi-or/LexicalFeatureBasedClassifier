from typing import Union
from .tanda_dataset import TandaDataset
from .feature_dataset import FeatureDataset

Dataset = Union[
    TandaDataset,
    FeatureDataset,
]