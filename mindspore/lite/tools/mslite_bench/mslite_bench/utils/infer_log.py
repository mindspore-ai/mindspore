"""
log for mslite bench
"""
import logging
from functools import wraps


def singleton(cls):
    """singleton decorator function"""
    instances_ = {}
    @wraps(cls)
    def _get_instances(*args, **kwargs):
        if cls not in instances_:
            instances_[cls] = cls(*args, **kwargs)
        return instances_.get(cls, None)

    return _get_instances


@singleton
class InferLogger:
    """
    logger for mslite bench, with singleton decorated
    """
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.logger = self._create_logger()

    def info(self, msg: str):
        """info level log"""
        return self.logger.info(msg)

    def debug(self, msg: str):
        """debug level log"""
        return self.logger.debug(msg)

    def warning(self, msg: str):
        """warning level log"""
        return self.logger.warning(msg)

    def error(self, msg: str):
        """error level log"""
        return self.logger.error(msg)

    def _create_logger(self):
        """create logger for mslite bench"""
        logger = logging.getLogger('[EASY_INFER]')
        log_format = '%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format,
                                      datefmt='%m/%d %I:%M:%S %p')
        if self.file_path is not None:
            file_handler = logging.FileHandler(self.file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

        return logger
