from pathlib import Path

import re
import os
import gc
import joblib
import optuna
import random
import torch
import pytorch_lightning as pl


from shutil import copyfile
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.loggers.neptune import NeptuneLogger

from src.training.model import UTimeModel
from src.training.dataset import DSSDataModule

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from functools import partial
from multiprocessing import Manager


MAX_EPOCHS = 450
RAND_SEED = random.randint(0, pow(2, 16))
TRAINING_NAME = "utime_2023-12-05 12:00:00"
TRAINING_DIR = Path("data") / "training" / TRAINING_NAME


def objective(trial: optuna.trial.Trial):
    TRIAL_DIR = TRAINING_DIR / f"trial_{trial.number}"
    config = {
        "try": trial.suggest_int("try", low=0, high=100, step=1),
        "channels_size": trial.suggest_int("channels_size", low=8, high=1000),
        "network_depth": trial.suggest_int("network_depth", low=1, high=10),
        "in_channels": 2,
        "kernel_size": trial.suggest_int("kernel_size", low=1, high=10),
        "dilation": trial.suggest_int("dilation", low=1, high=10),
        "lr": trial.suggest_float("lr", low=1e-6, high=1e-2, log=True),
        "dropout": trial.suggest_float("dropout", low=0, high=1),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
        "lr_scheduler": trial.suggest_categorical(
            "lr_scheduler", ["CLR", "ROP", "None"]
        ),
        # "momentum": trial.suggest_float("momentum", low=0.75, high=1),
        "momentum": 0.9,
        "clr_step_size_up": 4 * 10,
        "seed": random.randint(0, pow(2, 16)),
    }

    config["filters"] = [
        config["channels_size"] * 2**i for i in range(config["network_depth"])
    ]
    config["maxpool_kernels"] = [
        4 + 2 * i for i in reversed(range(config["network_depth"]))
    ]

    model = UTimeModel(**config)

    seed_everything(config["seed"], True)

    # Configure batch size
    if config["channels_size"] == 64:
        config["batch_size"] = 32
    elif config["channels_size"] == 16 or config["channels_size"] == 8:
        config["batch_size"] = 128
    else:
        config["batch_size"] = 64

    dm = DSSDataModule(batch_size=config["batch_size"])

    print(
        f"Start trial {trial.number}, trial_dir={TRIAL_DIR}",
        flush=True,
    )
    print("Using config", config, flush=True)

    early_stop_callback = EarlyStopping(
        monitor="validation/loss_smooth",
        min_delta=0.00,
        patience=50,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="validation/loss_smooth",
        verbose=True,
        mode="min",
        dirpath="file://" + str(TRIAL_DIR / "checkpoints"),
    )

    optuna_prunning_callback = PyTorchLightningPruningCallback(
        trial,
        monitor="validation/loss_smooth",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [
        early_stop_callback,
        checkpoint_callback,
        optuna_prunning_callback,
        lr_monitor,
    ]

    neptune_logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_KEY"],
        project=os.environ["NEPTUNE_PROJECT"],
        tags=[TRAINING_NAME, "demo"],
        log_model_checkpoints=False,
    )

    neptune_logger.log_model_summary(model=model, max_depth=-1)
    neptune_logger.experiment["model/description"] = str(model)

    NEPTUNE_RUN_ID = re.sub(".*/", "", neptune_logger.experiment.get_url())

    print(
        f"Start trial {trial.number}, trial_dir={TRIAL_DIR}, neptune_id={NEPTUNE_RUN_ID}",
        flush=True,
    )
    print("Using config", config, flush=True)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=[neptune_logger],
        log_every_n_steps=2,
        accelerator="gpu",
        precision="16-mixed",
    )

    should_prune = False
    try:
        trainer.fit(model, dm)
    except optuna.TrialPruned as error:
        print("optuna.TrialPruned caught!", error)
        should_prune = True

    best_score = checkpoint_callback.state_dict()["best_model_score"]

    neptune_logger.experiment["training/best_val_loss"] = best_score
    neptune_logger.experiment.wait()
    neptune_logger.experiment.stop()

    copyfile(
        checkpoint_callback.best_model_path[7:],
        TRIAL_DIR / "checkpoints" / "best-checkpoint.ckpt",
    )

    if should_prune:
        raise optuna.exceptions.TrialPruned()

    return best_score


def load_study(search_space=None, use_pruner=False):
    os.system(f"mkdir -p '{TRAINING_DIR}'")

    if search_space is None:
        sampler = optuna.samplers.TPESampler(
            seed=RAND_SEED, multivariate=True, group=True, n_startup_trials=10
        )
    else:
        sampler = optuna.samplers.GridSampler(search_space)

    if use_pruner:
        pruner = optuna.pruners.PercentilePruner(
            percentile=50,
            n_startup_trials=10,
            n_warmup_steps=4 * 10,
            interval_steps=5,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        storage=f"sqlite:///{TRAINING_DIR / 'optuna.db'}",
        study_name=TRAINING_NAME,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
    )

    return study


def cleanup_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example of grid search space
    search_space = {
        "try": [1, 2, 3],
        "channels_size": [8],
        "network_depth": [2],
        "kernel_size": [3],
        "dilation": [2],
        "lr": [1e-3],
        "dropout": [0.25],
        "optimizer": ["Adam"],
        "lr_scheduler": ["ROP"],
    }

    study = load_study(search_space=search_space)
    study

    study.optimize(
        objective,
        n_trials=1,
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=[cleanup_callback],
        n_jobs=1,
    )

    print("Best study params", study.best_params, "giving best value", study.best_value)
