import logging
import os
from datetime import datetime
from accelerate import Accelerator
import multiprocessing

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, accelerator=None, log_file=None, console_level=logging.DEBUG, file_level=logging.INFO):
        if self._initialized:
            return
        self._initialized = True

        self.accelerator = accelerator or Accelerator()
        self.logger = logging.getLogger('uncomp_logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - [Process %(process_info)s] - %(levelname)s - %(message)s')

            # Console Handler
            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(console_level)
            self.console_handler.setFormatter(formatter)
            self.logger.addHandler(self.console_handler)

            # File Handler
            if log_file is None:
                log_dir = 'logs'
                current_date = datetime.now().strftime("%Y%m%d")
                daily_log_dir = os.path.join(log_dir, current_date)
                if not os.path.exists(daily_log_dir):
                    os.makedirs(daily_log_dir)
                log_file = os.path.join(daily_log_dir, f'uncomp_{datetime.now().strftime("%H%M%S")}_{self.accelerator.process_index}.log')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_process_info(self):
        return f"{self.accelerator.process_index}/{self.accelerator.num_processes}"

    def log(self, level, message):
        process_info = self.get_process_info()
        self.logger.log(level, message, extra={'process_info': process_info})
        
    def debug(self, message):
        self.log(logging.DEBUG, message)

    def info(self, message):
        self.log(logging.INFO, message)

    def warning(self, message):
        self.log(logging.WARNING, message)

    def error(self, message):
        self.log(logging.ERROR, message)

    def critical(self, message):
        self.log(logging.CRITICAL, message)

    def set_console_level(self, level):
        """动态修改 console_handler 的日志级别"""
        self.console_handler.setLevel(level)
    
# Usage example
if __name__ == "__main__":
    accelerator = Accelerator()
    logger = Logger(accelerator)
    logger.info("This message includes Accelerate process information")