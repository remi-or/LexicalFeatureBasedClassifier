# General imports
from typing import Union

# Output types import
from .classification import ClassificationOutput
from .top_ranking import TopRankingOutput

# Main output type
Output = Union[
    ClassificationOutput,
    TopRankingOutput,
]

#############################################################################
# All outputs must implement a __len__ function that returns the batch size #
#############################################################################