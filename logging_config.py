import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.DEBUG):
    """
    Configure application-wide logging and ensure all logs/errors are printed to both file and stdout
    
    Args:
        log_level: The minimum log level to capture (default: DEBUG)
    
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, "app.log")
    
    # Formatter for consistent log formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Rotating file handler (10 MB per file, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Stream handler to print all logs/errors to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove all existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add both handlers: file and stream
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    return root_logger
def get_logger(name):
    """
    Get a logger for a specific module
    
    Args:
        name: Usually __name__ from the calling module
        
    Returns:
        A logger instance with the specified name
    """
    return logging.getLogger(name) 