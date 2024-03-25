import logging
from os import path as osp

from asdmotion.utils import RESOURCES_ROOT


def init_logger(log_name, log_path=None):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_path is not None:
        fh = logging.FileHandler(osp.join(log_path, f'{log_name}.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info(f'Initialization Success: {log_name}')
    return logger


class SingletonLogManager:
    _app_logger = None

    @property
    def APP_LOGGER(self) -> _app_logger:
        if SingletonLogManager._app_logger is None:
            SingletonLogManager._app_logger = init_logger(log_name='application', log_path=osp.join(RESOURCES_ROOT, 'logs'))
        return SingletonLogManager._app_logger


LogManager = SingletonLogManager()
