"""
The ``configuration`` module loads in the config specified by ``listener/config.ini``
and optionally overwrites values specified in the user config ``listener/usr.config.ini``
if it exists. The resultant config is stored as a module attribute and ``Config`` object that
is accessible via ::

    from listener.config import conf

"""

import configparser
from pathlib import Path, PurePath


DEFAULT_CONFIG = PurePath.joinpath(Path.cwd(), "listener/config.ini")
USER_CONFIG = PurePath.joinpath(Path.cwd(), "listener/usr.config.ini")


def _load_configuration(path: str, user_config_path: str = None):
    """
    Loads in the configuration

    :param str path: 
        The path to the default .ini configuration file
    
    :param str user_config_path:
        Optional path to a user .ini configuration file. If it is provided, it will
        be used to overwrite the values from the configuration file provided in the first
        parameter. Be sure that the user config file has the same structure and naming
        as the default .ini file.

    :return:
        config object
    """
    # Load default config
    default_config = configparser.ConfigParser()
    default_config.read(path)

    # Load user config
    if user_config_path and Path.is_file(user_config_path):
        config = configparser.ConfigParser()
        config.read([path, user_config_path])

    return config

# Load config
conf = _load_configuration(path=DEFAULT_CONFIG, user_config_path=USER_CONFIG)