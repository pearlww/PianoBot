
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss, _Loss
import config

class TransformerLoss(CrossEntropyLoss):
    def __init__(self, ignore_index=config.pad_token, reduction='mean') -> None:
        self.reduction = reduction
        self.ignore_index = ignore_index
        super().__init__(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: [B, T, V]
            target: [B , T]
        Returns:
            cross entropy: [1]
        """
        
        target = target.to(torch.long)
        mask = (target != self.ignore_index).to(input.device, dtype=torch.long)
        
        not_masked_length = mask.to(torch.int).sum()
        input = input.permute(0, -1, -2) # switch T and V

        _loss = super().forward(input, target)
        #_loss = self.forward(input, target)

        _loss *= mask.to(_loss.dtype)
        return _loss.sum() / not_masked_length

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward(input, target)

