import logging
import os
from datetime import datetime

class Logger:
    def __init__(self):
        # 创建logs目录
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 设置日志文件名
        log_file = os.path.join(
            log_dir, 
            f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('emotion_recognition')
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message, exc_info=True):
        self.logger.error(message, exc_info=exc_info)
    
    def debug(self, message):
        self.logger.debug(message) 