import os
import importlib
import argparse


def add(CONFIG, key_, default):
    value = os.environ.get(key_, default)
    if value:
        CONFIG[key_] = value


def _parse_args(CONFIG):
    parser = argparse.ArgumentParser()
    for key_ in CONFIG:
        parser.add_argument(f'--{key_.lower()}', default=CONFIG[key_])

    args, _ = parser.parse_known_args()
    return args


def _update_with_argparse(CONFIG):
    args = _parse_args(CONFIG)
    for arg in vars(args):
        CONFIG[arg.upper()] = getattr(args, arg)


def _update_with_os_environ(CONFIG):
    for _, key_ in enumerate(os.environ):
        if key_ in CONFIG:
            CONFIG[key_] = os.environ.get(key_)


ENV = os.environ.get('PYTHON_ENV', 'default')
CONFIG = importlib.import_module(f'style_transfer.config.{ENV}').CONFIG
_update_with_argparse(CONFIG)
_update_with_os_environ(CONFIG)
locals().update(CONFIG)
