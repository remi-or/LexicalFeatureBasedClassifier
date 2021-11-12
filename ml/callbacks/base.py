from typing import Union, Optional, Any
from torch.optim import Optimizer
import torch

from ..logger import Logger


Model = Any


class Caller:

    def __init__(
        self,
        savepath : str,
    ) -> None:
        self.early_stopping = {'on' : False}
        self.savepath = savepath

    def save(
        self,
        model : Model,
        optimizer : Optional[Optimizer] = None,
    ) -> None:
        torch.save(model.state_dict(), self.savepath + 'model.pt')
        if optimizer is not None:
            torch.save(optimizer.state_dict(), self.savepath + 'optimizer.pt')

    def load(
        self,
        model : Model,
        optimizer : Optional[Optimizer] = None,
    ) -> None:
        model.load_state_dict(torch.load(self.savepath + 'model.pt'))
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(self.savepath + 'optimizer.pt'))

    def add_early_stopping(
        self,
        patience : int,
        metric_name : str,
        warmup : int = 0,
        restore_best_weights : bool = True,
    ) -> None:
        self.early_stopping['on'] = True
        self.early_stopping['patience'] = patience
        self.early_stopping['metric_name'] = metric_name
        self.early_stopping['waited'] = patience + warmup
        self.early_stopping['best'] = None
        self.early_stopping['restore_best_weights'] = restore_best_weights,

    def _early_stopping(
        self,
        model : Model,
        optimizer : Optimizer,
        logger : Logger,
    ) -> bool:
        # Check if the callback is on
        if not self.early_stopping['on']:
            return False
        # Retrieve the metric
        metric_name = self.early_stopping['metric_name'] 
        metric = logger.metrics[metric_name]
        # Find the last measure
        last = metric[-1]
        # Retrieve best
        best = self.early_stopping['best']
        if best is None:
            self.early_stopping['best'] = last
        # Compare
        else:
            if metric.best(best, last) != best:
                self.early_stopping['best'] = last
                self.early_stopping['waited'] = max(self.early_stopping['patience'], self.early_stopping['waited'])
                if self.early_stopping['restore_best_weights']:
                    self.save(model, optimizer)
            else:
                self.early_stopping['waited'] -=1
        return self.early_stopping['waited'] == 0

    def __call__(
        self,
        model : Model,
        optimizer : Optimizer,
        logger : Logger,
    ) -> bool:
        stopping_early = self._early_stopping(model, optimizer, logger)
        return stopping_early

    def restore(
        self,
        model : Model,
        optimizer : Optimizer,
    ) -> None:
        if self.early_stopping['on'] and self.early_stopping['restore_best_weights']:
            self.load(model, optimizer)