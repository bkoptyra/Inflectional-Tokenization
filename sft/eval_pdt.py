from utils.run_pdt_eval import main


from datetime import timedelta
import os
from copy import deepcopy


import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator, InitProcessGroupKwargs


config_path = "conf_sft_pdt"
config_name = "base-config"


@hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name
)
def run(cfg: DictConfig):
    training_args = cfg.training

    print(training_args)
    timeout_time = timedelta(seconds=int(cfg.custom.timeout))
    kwargs = InitProcessGroupKwargs(timeout=timeout_time)
    accelerator = Accelerator(
        device_placement=False,
        log_with="wandb",
        kwargs_handlers=[
            kwargs
        ],
    )
    accelerator.init_trackers(
        project_name="sft_pdt_eval",
        init_kwargs={
            "wandb": {
                "tags": ["sd", cfg.model.model_type],
                "entity": "entity",
                "name": "name"
            }
        },
    )

    cfg_primitive = OmegaConf.to_container(cfg)

    accelerator.log(cfg_primitive)

    if cfg_primitive['data']['dataset_special_type'] == 'pdt':
        seed = int(cfg_primitive['training']['seed'])
        data_seed = int(cfg_primitive['training']['data_seed'])
        how_many = cfg_primitive['custom']['multiple_seeds']
        old_output_dir = cfg_primitive['training']['output_dir']
        for i in range(how_many):
            cfg_primitive['training']['seed'] = seed + i
            cfg_primitive['training']['data_seed'] = data_seed + i
            cfg_primitive['custom']['prefix'] = str(seed + i)
            if '/' != cfg_primitive['training']['output_dir'][-1]:
                cfg_primitive['training']['output_dir'] += '/'
            cfg_primitive['training']['output_dir'] += cfg_primitive['custom']['prefix']
            cfg_prim = deepcopy(cfg_primitive)
            main(
                accelerator=accelerator,
                cfg_primitive=cfg_prim,
            )
            cfg_primitive['training']['output_dir'] = old_output_dir
    else:
        main(
            accelerator=accelerator,
            cfg_primitive=cfg_primitive,
        )


if __name__ == '__main__':
    run()
