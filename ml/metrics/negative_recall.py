from .base import Metric
from ..outputs import Output, ClassificationOutput, TopRankingOutput


class NegativeRecall(Metric):

    nature = 'Negative recall'
    best = max

    @classmethod
    def compute(
        cls,
        outputs: Output,
    ) -> float:
        """
        Computes negative recall from (outputs).
        """
        # Classification
        if isinstance(outputs, ClassificationOutput):
            predictions = outputs.X if outputs.argmaxed else outputs.X.argmax(1)
            correct = predictions.eq(outputs.Y).double() 
            positive = predictions.eq(1).double()
            tn = correct.mul(1 - positive).sum().item()
            fp = (1 - correct).mul(positive).sum().item()
            return tn / (tn + fp) if tn != 0 else 0.
        else:
            cls.unsupported_output_types(type(outputs), cls.nature)
