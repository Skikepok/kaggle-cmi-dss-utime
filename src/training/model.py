import time
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from pytorch_lightning.utilities import grad_norm

from src.training.loss import DiceLoss
from src.config import VAL_LOSS_SMOOTH_PARAM


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.5
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.padding = (
            self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1
        ) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True,
            ),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels),
        )
        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(
        self,
        filters=[16, 32, 64, 128],
        in_channels=5,
        maxpool_kernels=[10, 8, 6, 4],
        kernel_size=5,
        dilation=2,
        dropout=0.5,
    ):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernels = maxpool_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        assert len(self.filters) == len(
            self.maxpool_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied maxpool kernels ({len(self.maxpool_kernels)})!"

        self.depth = len(self.filters)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(
                        in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dilation=self.dilation,
                        dropout=self.dropout,
                    ),
                    ConvBlock(
                        in_channels=self.filters[k],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dilation=self.dilation,
                        dropout=self.dropout,
                    ),
                )
                for k in range(self.depth)
            ]
        )

        self.maxpools = nn.ModuleList(
            [nn.MaxPool1d(self.maxpool_kernels[k]) for k in range(self.depth)]
        )

        self.bottom = nn.Sequential(
            ConvBlock(
                in_channels=self.filters[-1],
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            ),
            ConvBlock(
                in_channels=self.filters[-1] * 2,
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            ),
        )

    def forward(self, x):
        shortcuts = []
        for layer, maxpool in zip(self.blocks, self.maxpools):
            z = layer(x)
            shortcuts.append(z)
            x = maxpool(z)

        # Bottom part
        encoded = self.bottom(x)

        return encoded, shortcuts


class Decoder(nn.Module):
    def __init__(
        self,
        filters=[128, 64, 32, 16],
        upsample_kernels=[4, 6, 8, 10],
        in_channels=256,
        out_channels=5,
        kernel_size=5,
        dropout=0.5,
    ):
        super().__init__()
        self.filters = filters
        self.upsample_kernels = upsample_kernels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.dropout = dropout
        assert len(self.filters) == len(
            self.upsample_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied upsample kernels ({len(self.upsample_kernels)})!"
        self.depth = len(self.filters)

        self.upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=self.upsample_kernels[k]),
                    ConvBlock(
                        in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dropout=self.dropout,
                    ),
                )
                for k in range(self.depth)
            ]
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(
                        in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dropout=self.dropout,
                    ),
                    ConvBlock(
                        in_channels=self.filters[k],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dropout=self.dropout,
                    ),
                )
                for k in range(self.depth)
            ]
        )

    def forward(self, z, shortcuts):
        for upsample, block, shortcut in zip(
            self.upsamples, self.blocks, shortcuts[::-1]
        ):
            z = upsample(z)

            padding = shortcut.shape[2] - z.shape[2]
            z = F.pad(z, (0, padding, 0, 0))

            z = torch.cat([shortcut, z], dim=1)
            z = block(z)

        return z


class UTimeModel(pl.LightningModule):
    def __init__(
        self,
        filters=None,
        in_channels=None,
        maxpool_kernels=None,
        kernel_size=None,
        dilation=None,
        lr=None,
        dropout=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.normalizer = nn.BatchNorm1d(self.hparams.in_channels)

        self.encoder = Encoder(
            filters=self.hparams.filters,
            in_channels=self.hparams.in_channels,
            maxpool_kernels=self.hparams.maxpool_kernels,
            kernel_size=self.hparams.kernel_size,
            dilation=self.hparams.dilation,
            dropout=self.hparams.dropout,
        )
        self.decoder = Decoder(
            filters=self.hparams.filters[::-1],
            upsample_kernels=self.hparams.maxpool_kernels[::-1],
            in_channels=self.hparams.filters[-1] * 2,
            kernel_size=self.hparams.kernel_size,
            dropout=self.hparams.dropout,
        )

        self.final_conv = nn.Conv1d(
            in_channels=self.hparams.filters[0],
            out_channels=2,
            kernel_size=1,
            bias=True,
        )
        nn.init.xavier_uniform_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

        self.softmax = nn.Softmax(dim=1)

        self.loss = DiceLoss()

    def forward(self, x):
        # Reshape
        x = x.permute(0, 2, 1)

        # Normalize inputs
        z = self.normalizer(x)

        # Run through encoder
        z, shortcuts = self.encoder(z)

        # Run through decoder
        z = self.decoder(z, shortcuts)

        # Run through 1x1 conv to collapse channels
        z = self.final_conv(z)

        # Run softmax
        z = self.softmax(z)

        return z

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)

        self.log_dict(norms)

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
            )

        res = {"optimizer": self.optimizer}

        if self.hparams.lr_scheduler == "CLR":
            lr_scheduler = CyclicLR(
                optimizer=self.optimizer,
                base_lr=self.hparams.lr / 10.0,
                max_lr=self.hparams.lr,
                step_size_up=self.hparams.get("clr_step_size_up", 5),
                mode="triangular",
                cycle_momentum=self.hparams.optimizer != "Adam",
                verbose=False,
            )
            res["lr_scheduler"] = {
                "name": "CyclicLR",
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        elif self.hparams.lr_scheduler == "ROP":
            lr_scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode="min",
                factor=self.hparams.get("rop_factor", 0.5),
                patience=25,
                min_lr=1e-6,
                verbose=False,
            )

            res["lr_scheduler"] = {
                "name": "ReduceLROnPlateau",
                "scheduler": lr_scheduler,
                "strict": True,
                "monitor": "validation/loss_smooth",
            }

        return res

    def save_plot_prediction(self, name, X, y, preds, metadatas):
        """
        Plot a random event and upload it to neptune for debugging of the training
        """
        # Select random batch item
        _idx = np.random.randint(0, X.shape[0])
        event_id, (start_index) = metadatas[_idx]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f"{event_id=}, {start_index=}", fontsize=16)
        ax.plot(X[_idx, ::5, 0].cpu(), color="black")
        ax.plot(y[_idx, ::5].cpu() * 50, color="green")
        ax.plot(preds[_idx, 1, ::5].cpu() * 50, color="red")
        fig.tight_layout()
        plt.close(fig)

        self.logger.experiment[name].append(fig)

    def setup(self, stage):
        self.val_loss_smooth = None

    def training_step(self, batch, batch_idx):
        inputs, targets, metadata = batch

        preds = self(inputs)

        loss = self.loss(preds, targets)

        self.log(
            "training/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss, "predicted": preds, "targets": targets}

    def on_validation_epoch_start(self):
        self.val_loss = []

    def validation_step(self, batch, batch_idx):
        inputs, targets, metadata = batch

        preds = self(inputs)

        loss = self.loss(preds, targets)

        self.log(
            "validation/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        # Log a resulting image
        self.save_plot_prediction("validation/plot", inputs, targets, preds, metadata)

        self.val_loss.append(loss.cpu())

        return {"loss": loss, "predicted": preds, "targets": targets}

    def on_validation_epoch_end(self):
        val_loss = np.mean(self.val_loss)

        # Compute smoother version of the validation loss as the original one
        # frequently reaches newer lows by "luck"
        if self.val_loss_smooth is None:
            self.val_loss_smooth = val_loss
        else:
            self.val_loss_smooth = (
                self.val_loss_smooth * (1 - VAL_LOSS_SMOOTH_PARAM)
                + val_loss * VAL_LOSS_SMOOTH_PARAM
            )

        self.log(
            "validation/loss_smooth",
            self.val_loss_smooth,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_index):
        inputs, targets, metadata = batch

        preds = self(inputs)

        loss = self.loss(preds, targets)

        self.log(
            "test/loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )

        # Log a resulting image
        self.save_plot_prediction("test/plot", inputs, targets, preds, metadata)

        return {"loss": loss, "predicted": preds, "targets": targets}
