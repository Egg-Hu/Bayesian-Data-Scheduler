from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from openrlhf.models.utils import masked_mean


class BatchCrossEntropyLoss(nn.Module):
    """
    Unreduced SFT Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(reduction="none", ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, sequence_reduce="mean") -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        unreduced_loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())
        if sequence_reduce=='mean':
            batch_loss = unreduced_loss.sum(1)/(shift_labels!=self.IGNORE_INDEX).int().sum(1)
        elif sequence_reduce=='sum':
            batch_loss = unreduced_loss.sum(1)
        else:
            raise ValueError("sequence_reduce can be mean or sum")
        return batch_loss
        
  
