"""
This script fits a neural network to the output of an auditory filterbank.
"""
import datetime
from datetime import timedelta
import fire
import h5py
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import time
import torch

import student
import teacher

MAX_EPOCHS = 100
SAMPLES_PER_EPOCH = 8000

def run(sav_dir, domain, arch, init_id, batch_size, job_id):
    # Print header
    start_time = int(time.time())
    print("Job ID: " + str(job_id))
    print(str(datetime.datetime.now()) + " Start.")
    print(__doc__ + "\n")
    print("\n".join([sav_dir, domain, arch, str(init_id)]) + "\n")

    # Print version numbers
    for module in [h5py, np, pd, pl, torch]:
        print("{} version: {:s}".format(module.__name__, module.__version__))
    print("")
    sys.stdout.flush()

    # Create model directory
    model_dir = os.path.join(sav_dir, "models", domain)
    model_sav_path = os.path.join(model_dir, str(job_id))
    os.makedirs(model_sav_path, exist_ok=True)
    pred_path = os.path.join(model_sav_path, "predictions.npy")

    # Initialize dataset
    if domain == "synth":
        dataset = teacher.CQTSineDataModule(domain=domain, batch_size=batch_size)
    else:
        dataset = teacher.SpectrogramDataModule(
            sav_dir=sav_dir, domain=domain, batch_size=batch_size)
    print(str(datetime.datetime.now()) + " Finished initializing dataset")

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = -1
        torch.multiprocessing.set_start_method("spawn")
    else:
        accelerator = "cpu"
        devices = "auto"
    print("Current device: ", accelerator)

    # Initialize model
    constructor = getattr(student, arch)
    spec = teacher.HYPERPARAMS[domain]
    model = constructor(spec)
    print(str(datetime.datetime.now()) + " Finished initializing model")

    # Setup checkpoints and Tensorboard logger
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_sav_path,
        monitor="val_loss",
        save_last=True,
        filename="best",
        save_weights_only=False,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(model_sav_path, "logs")
    )

    # Setup trainer
    steps_per_epoch = SAMPLES_PER_EPOCH // batch_size
    max_steps = steps_per_epoch * MAX_EPOCHS
    strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=MAX_EPOCHS,
        max_steps=max_steps,
        limit_train_batches=steps_per_epoch,
        callbacks=[checkpoint_cb],
        logger=tb_logger,
        max_time=timedelta(hours=12),
        strategy=strategy
    )    

    # Train
    trainer.fit(model, dataset)

    # Print elapsed time.
    print(str(datetime.datetime.now()) + " Success.")
    elapsed_time = time.time() - int(start_time)
    elapsed_hours = int(elapsed_time / (60 * 60))
    elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
    elapsed_seconds = elapsed_time % 60.0
    elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(
        elapsed_hours, elapsed_minutes, elapsed_seconds
    )
    print("Total elapsed time: " + elapsed_str + ".")


# Launch computation
if __name__ == "__main__":
    #save_dir = "/home/han/localdata/data/waspaa_data/SOL"
    #domain = "speech"
    #arch, init_id, batch_size, job_id = "Gabor1D", 0, 2, 1
    fire.Fire(run)
