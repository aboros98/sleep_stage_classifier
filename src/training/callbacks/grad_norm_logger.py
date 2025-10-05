from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import grad_norm


class GradNormLogger(Callback):
    """Callback to log gradient norms during training."""
    
    def __init__(
        self,
        norm_type: int = 2,
        log_all_layers_norm: bool = False,
        log_on_step: bool = True,
        log_on_epoch: bool = False
    ) -> None:
        """
        Initialize the GradNormLogger.
        
        Args:
            norm_type: Type of norm to compute (e.g., 2 for L2 norm).
            log_all_layers_norm: Whether to log norms for all layers individually.
            log_on_step: Whether to log on each training step.
            log_on_epoch: Whether to log on each epoch.
        """
        super().__init__()
        self.norm_type = norm_type
        self.log_all_layers_norm = log_all_layers_norm
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: Optional[object]
    ) -> None:
        """
        Log gradient norms before optimizer step.
        
        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being trained.
            optimizer: The optimizer (unused but required by callback signature).
        """
        norms = grad_norm(pl_module, norm_type=self.norm_type)

        if self.log_all_layers_norm:
            pl_module.log_dict(
                norms,
                on_step=self.log_on_step,
                on_epoch=self.log_on_epoch,
                prog_bar=False,
                logger=True
            )
        else:
            for key, value in norms.items():
                if key == "grad_2.0_norm_total":
                    pl_module.log(
                        "grad/global_norm",
                        value,
                        on_step=self.log_on_step,
                        on_epoch=self.log_on_epoch,
                        prog_bar=False,
                        logger=True
                    )
                    break
