from utils.run_pretrain import main


from datetime import timedelta
import os


from datasets import load_dataset, load_from_disk, DatasetDict
import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator, InitProcessGroupKwargs, DeepSpeedPlugin


config_path = "conf_pretrain"
config_name = "base-config"


@hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name
)
def run(cfg: DictConfig):
    training_args = cfg.training

    if 'resume_from_checkpoint' in training_args and 'checkpoint_on_tmpdir' in training_args and training_args['checkpoint_on_tmpdir']:
        training_args.resume_from_checkpoint = str(os.environ['TMPDIR']) \
            + training_args.resume_from_checkpoint

    model_args = cfg.model

    custom_args = cfg.custom
    if (
        'add_new_layers' in custom_args and
        custom_args.add_new_layers and
        'new_state_dict_path' in custom_args
    ):
        custom_args.new_state_dict_path = str(os.environ['TMPDIR']) \
            + custom_args.new_state_dict_path
        print(
            'Will add new layers.',
            'Freezing old layers:',
            custom_args.freeze_old_layers
        )

    deepspeed_plugin = DeepSpeedPlugin('pretraining/conf_pretrain/deepspeed/base_config.json')
    print(deepspeed_plugin)
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        log_with="wandb",
        kwargs_handlers=[
            InitProcessGroupKwargs(
                timeout=timedelta(minutes=int(custom_args.timeout))
            )
        ],
    )
    accelerator.init_trackers(
        project_name="pretrain",
        init_kwargs={
            "wandb": {
                "tags": ["sd", model_args.model_name_or_path],
                "entity": "entity",
                "name": "name"
            }
        },
    )

    cfg_primitive = OmegaConf.to_container(cfg)

    print(cfg_primitive)
    accelerator.log(cfg_primitive)

    if custom_args.save_to_disk_dataset:
        ds = load_from_disk(cfg.data.data_dir)
    else:
        ds = load_dataset(cfg.data.data_dir)

    if not custom_args.is_datasetdict:
        ds = DatasetDict({'train': ds})

    main(
        accelerator=accelerator,
        preprocessed_datasets=ds,
        cfg_primitive=cfg_primitive
    )


if __name__ == '__main__':
    run()
