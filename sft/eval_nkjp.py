from utils.run_nkjp_eval import main


from datetime import timedelta
import os
from copy import deepcopy


import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import get_dataset_config_names, get_dataset_split_names


from utils.prepare_dataset import get_train_data_path


config_path = "conf_sft_nkjp"
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
        project_name="sft_nkjp_eval",
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

    if cfg_primitive['data']['dataset_special_type'] == 'nkjp_cv':
        train_data_path = str(get_train_data_path(cfg_primitive['data']))
        configs = get_dataset_config_names(train_data_path)
        assert len(configs) == 1
        configs = configs[0]

        splits = get_dataset_split_names(train_data_path, configs)
        old_output_dir = cfg_primitive['training']['output_dir']
        for cv_test_split in splits:
            cfg_primitive['data']['cv_test_split'] = cv_test_split
            if '/' != cfg_primitive['training']['output_dir'][-1]:
                cfg_primitive['training']['output_dir'] += '/'
            cfg_primitive['training']['output_dir'] += str(cv_test_split)
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
