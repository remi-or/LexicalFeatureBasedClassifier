from .base import Metric
from .loss import Loss
from .accuracy import Accuracy
from .f1 import F1
from .auxiliary_accuracy import AuxiliaryAccuracy
from .mrr import MeanReciprocalRank
from .negative_recall import NegativeRecall

def parse(
    description : str,
) -> Metric:
    """
    Parses and returns a metric from a (description).
    """
    phase, nature = description.split(' ')
    is_training = phase in ['train', 'training', 'Train', 'Training']
    is_validation = phase in ['val', 'validation', 'Val', 'Validation']
    if nature in ['loss', 'Loss']:
        return Loss(is_training, is_validation)
    elif nature in ['acc', 'accuracy', 'Acc', 'Accuracy']:
        return Accuracy(is_training, is_validation)
    elif nature in ['aux_acc', 'auxiliary_accuracy', 'AuxAcc']:
        return AuxiliaryAccuracy(is_training, is_validation)
    elif nature in ['f1', 'F1']:
        return F1(is_training, is_validation)
    elif nature in ['mrr', 'MeanReciprocalRank', 'MRR']:
        return MeanReciprocalRank(is_training, is_validation)
    elif nature in ['nr', 'neg recall', 'negative_recall', 'Negative_recall']:
        return NegativeRecall(is_training, is_validation)
    else:
        raise(ValueError(f"Unknown metric nature {nature} that was parsed from {description}."))