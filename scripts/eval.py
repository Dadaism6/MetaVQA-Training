"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *
import wandb
import datetime
os.environ["WANDB__SERVICE_WAIT"] = "300"
def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # replace some settings in the used config
    parser.add_argument("--replace_cfg", nargs="+", help="replace some settings in the used config", default=None)
    parser.add_argument("--job_id", default=None, help="job id")
    # python train.py --cfg-path configs/cont_train.yaml \
    # --replace-cfg run_cfg.seed=1 run_cfg.local_rank=0 --job-id 1

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    args = parse_args()
    cfg = Config(args)

    torch.multiprocessing.set_start_method("spawn")
    job_id = now() if args.job_id is None else args.job_id
    WANDB_ENV_VAR = "WANDB_API_KEY"
    os.environ[WANDB_ENV_VAR] = "6048623673469aa435e5c18e4c00a653edc76a5c"
    wandb_cfg = cfg.wandb_cfg
    group_name = wandb_cfg.get("group_name", "you_forget_your_group_name!")
    project_name = wandb_cfg.get("project_name", "you_forget_your_project_name!")

    args.rank = init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    trial_name = "{}_{}_id{}".format(wandb_cfg.get("trial_name", "you_forget_your_trial_name!"), get_time_str(),
                                     args.rank)
    print("!!!!!!Rank is: {}".format(args.rank))
    wandb_run = wandb.init(
        id=trial_name,
        config=cfg or {},
        resume=True,
        reinit=True,
        # allow_val_change=True,
        group=group_name,
        project=project_name,
        entity="chendaduan",
        sync_tensorboard=True,
        save_code=False
    )
    cfg.pretty_print()

    # with torch.cuda.amp.autocast(dtype=torch.float32):
    task = tasks.setup_task(cfg)
    task.pass_wandb(wandb_run)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    # runner = RunnerBase(
    #     cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    # )
    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
