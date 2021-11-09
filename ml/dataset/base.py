from __future__ import annotations

from typing import Optional, List, Any, Tuple, Union
import numpy as np


Number = Union[int, float]


class Dataset:

    # The yield_order is the common attributes to all datasets in which the dataset is read-through.
    yield_order : List[Any]
    shuffles : int = 0

    def reset_randomness(self) -> None:
        self.shuffles = 0

    def shuffle(
        self,
        seed : Optional[int] = None,
    ) -> None:
        """
        Shuffles the .yield_order attribute of the dataset.
        A (seed) can be passed for reproducibilty's sake.
        """
        self.shuffles += 1
        seed = None if seed is None else seed + self.shuffles
        rng = np.random.default_rng(seed)
        rng.shuffle(self.yield_order)

    def get_number_of_batches(
        self,
        **dataset_kwargs,
    ) -> int:
        """
        Counts the number of batches returned when the .batches(**dataset_kwargs) method is called.
        """
        dataset_kwargs['shuffle'] = False
        number_of_batches = 0
        for _ in self.batches(**dataset_kwargs):
            number_of_batches += 1
        return number_of_batches