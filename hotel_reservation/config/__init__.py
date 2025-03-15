"""Configuration for Hotel Reservation.

Three configurations sources are merged in the following order of increasing
precedence.

1. the default config in config/default.yaml.
2. the config file named by the CONF_FILE environment variable.
3. the dictionary of keys and values specified in ENVIRONMENT_VARIABLES
   in dot notation. For example, `{"hotel_reservation.url": "value"}`.

"""
import json
import os

import yaml
from omegaconf import OmegaConf

# This has the lowest precedence
default_conf = OmegaConf.load('hotel_reservation/config/default.yaml')

# Load from file named in CONF_FILE=
tmp_conf = OmegaConf.create()
if os.environ.get('CONF_FILE', None):
    tmp_conf = OmegaConf.load(os.environ.get('CONF_FILE'))

# Also assume that ENVIRONMENT_VARIABLES contains a dictionary of key values.
custom_conf = OmegaConf.from_dotlist([
    f'{k}={v}' for k, v in json.loads(
        os.environ.get('ENVIRONMENT_VARIABLES', '{}')).items()
])

config = OmegaConf.merge(default_conf, tmp_conf, custom_conf)

with open('params.yaml', 'r', encoding='utf-8') as f:
    dvc_params = yaml.safe_load(f)