from prefigure.prefigure import get_all_args, push_wandb_config
import json
import os
import torch
import pytorch_lightning as pl
import random
import signal

import torchinfo
from pytorch_lightning import callbacks, loggers, accelerators, plugins, profilers, strategies
from pytorch_lightning.plugins.environments import SLURMEnvironment

from stable_audio_tools.data.dataset import create_dataloader_from_configs_and_args
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


SIGUSR1_TRIGGERED = False
SIGUSR1_SAVED = False

def signal_handler_usr1(sig, frame):
    print(f'Got SigUSR1. Stopping...', flush=True)
    global SIGUSR1_TRIGGERED
    SIGUSR1_TRIGGERED = True

print('Registering handler for SigUSR1...')
signal.signal(signal.SIGUSR1, signal_handler_usr1)

class SaveOnSigUSR1Callback(pl.Callback):
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        global SIGUSR1_TRIGGERED, SIGUSR1_SAVED
        if SIGUSR1_TRIGGERED and not SIGUSR1_SAVED:
            hpc_save_path = trainer._checkpoint_connector.hpc_save_path(trainer.default_root_dir)
            trainer.save_checkpoint(hpc_save_path)

            # Stop the trainer gracefully after saving the checkpoint
            trainer.should_stop = True

            SIGUSR1_TRIGGERED = False
            SIGUSR1_SAVED = True


def main():

    args = get_all_args()

    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    random.seed(seed)
    torch.manual_seed(seed)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_configs_and_args(model_config, args, dataset_config)

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        print(f'Loading pretrained model from {args.pretrained_ckpt_path}')
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))
    
    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
    
    training_wrapper = create_training_wrapper_from_config(model_config, model)
    rank_zero_only(torchinfo.summary)(training_wrapper, depth=7)

    wandb_logger = loggers.WandbLogger(project=args.name)
    wandb_logger.watch(training_wrapper, log="all", log_freq=500)

    exc_callback = ExceptionCallback()

    RUN_ID = args.id or os.environ.get("SLURM_JOB_ID")
    if RUN_ID:
        print(f'RUN_ID: {RUN_ID}')
        save_dir = os.path.join(args.save_dir, args.name, str(RUN_ID))
    elif args.save_dir and isinstance(wandb_logger.experiment.id, str):
        save_dir = os.path.join(args.save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id)
    else:
        save_dir = None

    print(f'Save dir: {save_dir}')

    ckpt_callback = callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=save_dir, save_last=True)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    push_wandb_config(wandb_logger, args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            strategy = strategies.DeepSpeedStrategy(
                stage=2,
                contiguous_gradients=True,
                overlap_comm=True,
                reduce_scatter=True,
                reduce_bucket_size=5e8,
                allgather_bucket_size=5e8,
                load_full_weights=True,
            )
        elif args.strategy == "ddp":
            strategy = strategies.DDPStrategy(
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
                static_graph=False,
                bucket_cap_mb=40,
            )
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto" 

    trainer_callbacks = [
        ckpt_callback,
        demo_callback,
        exc_callback,
        save_model_config_callback,
        SaveOnSigUSR1Callback(),
    ]

    if model_config["training"].get("grad_accum_schedule"):
        trainer_callbacks.append(
            callbacks.GradientAccumulationScheduler(
                dict(model_config["training"]["grad_accum_schedule"])
            )
        )

    trainer_plugins = []

    if SLURMEnvironment.detect():
        trainer_plugins.append(SLURMEnvironment(auto_requeue=False))


    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=trainer_callbacks,
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        default_root_dir=save_dir,
        gradient_clip_val=args.gradient_clip_val,
        plugins=trainer_plugins,
    )

    trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()