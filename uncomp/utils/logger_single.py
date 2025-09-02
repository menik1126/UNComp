import logging
import os
from datetime import datetime

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_file=None, console_level=logging.DEBUG, file_level=logging.INFO):
        if self._initialized:
            return
        self._initialized = True

        self.logger = logging.getLogger('uncomp_logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File Handler
            if log_file is None:
                log_dir = 'logs'
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

# 使用示例
if __name__ == "__main__":
    logger1 = Logger()
    logger2 = Logger()  # 这将返回与 logger1 相同的实例

    logger1.info("这条信息应该只出现一次")
    logger2.info("这条信息也应该只出现一次")