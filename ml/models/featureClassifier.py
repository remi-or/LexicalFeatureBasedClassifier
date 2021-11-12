from __future__ import annotations
from typing import Optional, List
import torch
from torch import nn
from torch.optim import Optimizer

from torch import Tensor

from ..callbacks import Callbacks
from ..module import Module
from ..dataset.feature_dataset import FeatureBatch, FeatureDataset
from ..outputs import ClassificationOutput
from ..logger import Logger
from ..dataset import Dataset
from ml import outputs



class FeatureClassifier(Module):

    def __init__(
        self,
        number_of_features : int,
        dropout : float = 0.,
        hidden_size : Optional[int] = None,
        weights : Optional[Tensor] = None,
        seed : Optional[int] = None,
        ) -> None:
        super(FeatureClassifier, self).__init__()
        self.number_of_features = number_of_features
        self.hidden_size = number_of_features if hidden_size is None else hidden_size
        self.dropout_rate = dropout
        self.seed = seed
        self.build(weights, seed)

    def build(
        self,
        weights : Optional[Tensor],
        seed : Optional[int] = None,
    ) -> None:
        self.linear_0 = nn.Linear(in_features=self.number_of_features, out_features=self.hidden_size)
        self.linear_1 = nn.Linear(in_features=self.hidden_size, out_features=2)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.loss = nn.CrossEntropyLoss(weights)
        self.init_weights(seed)

    def init_weights(
        self,
        seed : Optional[int] = None
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.normal_(self.linear_0.weight)
        nn.init.constant_(self.linear_0.bias, 0.01)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0.01)
    
    def forward(
        self,
        batch : FeatureBatch,
        with_loss : bool,
    ) -> ClassificationOutput:
        batch.to(self.device)
        X = batch.X
        X = self.linear_0(X)
        X = torch.relu(X)
        X = self.dropout(X)
        X = self.linear_1(X)
        X = torch.softmax(X, 1)
        loss = self.loss(X, batch.Y) if with_loss else None
        return ClassificationOutput(
            predictions=X,
            labels=batch.Y,
            loss=loss,
        )

    def fit(
        self,
        optimizer : Optimizer,
        logger : Logger,
        epochs : int,
        training_dataset : Dataset,
        validation_dataset : Optional[Dataset] = None,
        batch_size : int = 1,
        backpropagation_frequency : int = 1,
        pre_training_validation : bool = False,
        verbose : bool = False,
        callbacks : Optional[Callbacks] = None,
        seed : Optional[int] = None,
    ) -> None:
        super().fit(
            optimizer=optimizer,
            logger=logger,
            epochs=epochs,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            dataset_kwargs={
                'batch_size' : batch_size,
                'seed' : seed,
            },
            backpropagation_frequency=backpropagation_frequency,
            pre_training_validation=pre_training_validation,
            verbose=verbose,
            callbacks=callbacks,
        )

    def predict(
        self,
        dataset : FeatureDataset,
        batch_size : int = 1,
    ) -> List[int]:
        labels = []
        self.eval()
        with torch.no_grad():
            for batch in dataset.batches(batch_size=batch_size, shuffle=False):
                predictions = self.forward(batch=batch, with_loss=False).X
                argmaxed_predictions = predictions.argmax(1)
                list_of_predictions = list(argmaxed_predictions.detach().cpu().numpy())
                labels += list_of_predictions
        return labels

    def save(
        self,
        path : str,
    ) -> None:
        """
        Saves the whole model (architecture + weights + other attributes) at the pointed (path).
        """
        state_dict_plus = {}
        state_dict_plus['state_dict'] = self.state_dict()
        state_dict_plus['__init__'] = {
            'number_of_features' : self.number_of_features,
            'dropout' : self.dropout_rate,
            'hidden_size' : self.hidden_size,
            'weights' : None, # This is taken care of by the classic state_dict
            'seed' : self.seed,

        }
        torch.save(state_dict_plus, path)

    @staticmethod
    def load(
        path : str,
    ) -> FeatureClassifier:
        """
        Loads a model saved with the .save method at (path).
        """
        state_dict_plus = torch.load(path)
        instance = FeatureClassifier(**state_dict_plus['__init__'])
        instance.load_state_dict(state_dict_plus['state_dict'])
        return instance