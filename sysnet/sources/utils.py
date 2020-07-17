import logging
import os


def set_logger(log_path=None, level='info'):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
        
        
    credit: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py    
    """
    levels = {
        'info' : logging.INFO,
        'debug' : logging.DEBUG,
        'warning' : logging.WARNING,
        }

    logger = logging.getLogger()
    logger.setLevel(levels[level])

    if not logger.handlers:
        
        if log_path is not None:
            # Logging to a file
            log_dir = os.path.dirname(log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)                
                #logging.info(f"created {log_dir}")
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s',
                                                        datefmt='%m-%d %H:%M '))
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)