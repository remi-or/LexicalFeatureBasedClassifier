from .base import Metric
from ..outputs import Output, ClassificationOutput, TopRankingOutput


class Loss(Metric):

    nature = 'Loss'
    best = min

    @classmethod
    def compute(
        cls,
        outputs : Output,
    ) -> float:
        """
        Computes loss from (outputs).
        """
        # Classification or TopRanking
        if isinstance(outputs, ClassificationOutput) or isinstance(outputs, TopRankingOutput):
            return outputs.loss.item()
        else:
            cls.unsupported_output_types(type(outputs), cls.nature)