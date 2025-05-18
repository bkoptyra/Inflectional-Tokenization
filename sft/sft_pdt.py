from utils.run_sft import main


from datetime import timedelta
import os
from copy import deepcopy


import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator, InitProcessGroupKwargs, DeepSpeedPlugin
import torch.distributed as dist
from datasets import get_dataset_config_names, get_dataset_split_names


from utils.prepare_dataset import get_train_data_path


config_path = "conf_sft_pdt"
config_name = "base-config"


@hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name
)
def run(cfg: DictConfig):
    training_args = cfg.training

    if 'resume_from_checkpoint' in training_args:
        training_args.resume_from_checkpoint = str(os.environ['TMPDIR']) \
            + training_args.resume_from_checkpoint

    timeout_time = timedelta(seconds=int(cfg.custom.timeout))
    dist.init_process_group(backend="nccl", timeout=timeout_time)
    deepspeed_config = 'sft/conf_sft_pdt/deepspeed/base_config.json'
    deepspeed_plugin = DeepSpeedPlugin(deepspeed_config)
    kwargs = InitProcessGroupKwargs(timeout=timeout_time)
    accelerator = Accelerator(
        device_placement=False,
        deepspeed_plugin=deepspeed_plugin,
        log_with="wandb",
        kwargs_handlers=[
            kwargs
        ],
    )
    accelerator.init_trackers(
        project_name="sft_pdt",
        init_kwargs={
            "wandb": {
                "tags": ["sd", cfg.model.model_type],
                "entity": "entity",
                "name": "name"
            }
        },
    )

    cfg_primitive = OmegaConf.to_container(cfg)

    print(cfg_primitive)
    accelerator.log(cfg_primitive)

    if 'multiple_seeds' in cfg_primitive['custom'] and cfg_primitive['custom']['multiple_seeds']:
        seed = int(cfg_primitive['training']['seed'])
        data_seed = int(cfg_primitive['training']['data_seed'])
        how_many = int(cfg_primitive['custom']['multiple_seeds'])
        old_output_dir = cfg_primitive['training']['output_dir']
        old_s3_rclone_output_dir = cfg_primitive['custom']['s3_rclone_output_dir']
        for i in range(how_many):
            cfg_primitive['training']['seed'] = seed + i
            cfg_primitive['training']['data_seed'] = data_seed + i
            if '/' != cfg_primitive['training']['output_dir'][-1]:
                cfg_primitive['training']['output_dir'] += '/'
            cfg_primitive['training']['output_dir'] += str(seed + i)
            cfg_primitive['custom']['s3_rclone_output_dir']
            if '/' != cfg_primitive['custom']['s3_rclone_output_dir'][-1]:
                cfg_primitive['custom']['s3_rclone_output_dir'] += '/'
            cfg_primitive['custom']['s3_rclone_output_dir'] += str(seed + i)
            cfg_prim = deepcopy(cfg_primitive)
            main(
                accelerator=accelerator,
                cfg_primitive=cfg_prim,
                deepspeed_config=deepspeed_config,
            )
            cfg_primitive['training']['output_dir'] = old_output_dir
            cfg_primitive['custom']['s3_rclone_output_dir'] = old_s3_rclone_output_dir
    else:
        main(
            accelerator=accelerator,
            cfg_primitive=cfg_primitive,
            deepspeed_config=deepspeed_config,
        )


if __name__ == '__main__':
    run()
