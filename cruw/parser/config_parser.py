import os
import json

from cruw.config_classes import HumanAnnoConfig, Loc3DCamConfig


def load_loc3d_cam_config_dict(config_name: str) -> Loc3DCamConfig:
    """
    Create a Loc3DCamConfig class for CRUW evaluation.
    The config file is located in 'cruw/eval/loc3d_cam/configs' folder.
    :param config_name: Name of configuration
    :return: Loc3DCamConfig
    """

    # check if config exists
    assert os.path.exists(config_name), 'Configuration {} not found'.format(config_name)

    # load config file to Loc3DCamConfig
    with open(config_name, 'r') as f:
        data = json.load(f)
    cfg = Loc3DCamConfig.initialize(data)

    return cfg


def load_human_anno_config_dict(config_name: str) -> HumanAnnoConfig:
    """
    Create a annotation class for CRUW label.
    The config file is located in 'cruw/eval/loc3d_cam/configs' folder.
    :param config_name: Name of configuration
    :return: Annotation
    """

    # check if config exists
    assert os.path.exists(config_name), 'Configuration {} not found'.format(config_name)

    # load config file to Loc3DCamConfig
    with open(config_name, 'r') as f:
        data = json.load(f)
    cfg = HumanAnnoConfig.initialize(data)

    return cfg
