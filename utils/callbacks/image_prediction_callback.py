import torch
import wandb
import pytorch_lightning as pl


class ImagePredictionCallback(pl.Callback):
    """
    This: https://github.com/wandb/examples/blob/master/colabs/pytorch-lightning/Supercharge_your_Training_with_Pytorch_Lightning_%2B_Weights_%26_Biases.ipynb
    """

    def __init__(self, samples: torch, batch_size: int = 16):
        super(ImagePredictionCallback, self).__init__()
        self.samples = samples
        self.batch_size = batch_size

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        samples = self.samples.to(pl_module.device)
        logits = pl_module(samples)
        y_pred = torch.argmax(logits, 1)

        trainer.logger.experiment.log(
            {
                "examples": [wandb.Image(x, caption=f'Label: {y}, predicted: {x}') for x, pred, y in
                             zip(samples, y_pred, self.samples)],
                "global_step": trainer.global_step
            })
