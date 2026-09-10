import logging
import os
import sys

def setup_logger(name="TehlikeSistemi"):
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger= logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler= logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler= logging.FileHandler('logs/sistem.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)



    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger 

logger= setup_logger()

