"""
Logger Utility - Standardized logging for the auto trading bot

Provides:
1. Consistent logging format across all system components
2. Log level configuration based on settings
3. Options for console and file logging
4. Contextual information in logs (module, timestamp, log level)
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import traceback

# Add project root to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing from config, but use defaults if not available
try:
    from config.settings import (
        LOG_LEVEL, 
        LOG_FORMAT, 
        LOG_DATE_FORMAT, 
        LOG_FILE, 
        MAX_LOG_SIZE,
        LOG_BACKUP_COUNT,
        CONSOLE_LOGGING_ENABLED,
        FILE_LOGGING_ENABLED
    )
except ImportError:
    # Default settings if config is not available
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_FILE = "logs/auto_trading.log"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT = 5
    CONSOLE_LOGGING_ENABLED = True
    FILE_LOGGING_ENABLED = True

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Map string log levels to logging constants
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def get_logger(name=None):
    """
    Get a configured logger instance.
    
    Args:
        name (str, optional): Logger name, typically __name__ of the module.
            If None, returns the root logger.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Convert string log level to logging constant
    log_level = LOG_LEVELS.get(LOG_LEVEL.upper(), logging.INFO)
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Only configure the logger if it hasn't been configured
    if not logger.handlers:
        logger.setLevel(log_level)
        logger.propagate = False  # Prevent double logging in root logger
        
        # Create formatters
        formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        
        # Add console handler if enabled
        if CONSOLE_LOGGING_ENABLED:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if FILE_LOGGING_ENABLED:
            # Use RotatingFileHandler to limit file size
            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=MAX_LOG_SIZE,
                backupCount=LOG_BACKUP_COUNT
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
    
    return logger

def log_exception(logger, exception, context=""):
    """
    Log an exception with full traceback.
    
    Args:
        logger (logging.Logger): Logger instance
        exception (Exception): The exception to log
        context (str, optional): Additional context information
    """
    error_msg = f"{context} - {type(exception).__name__}: {str(exception)}"
    stack_trace = "".join(traceback.format_exception(
        type(exception), exception, exception.__traceback__))
    
    logger.error(f"{error_msg}\n{stack_trace}")

class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds contextual information to log messages.
    Useful for adding trade IDs, user IDs, or other context.
    """
    
    def __init__(self, logger, extra=None):
        """
        Initialize the adapter with a logger and extra context.
        
        Args:
            logger (logging.Logger): The logger to adapt
            extra (dict, optional): Extra contextual information to add to all logs
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Add contextual information to the log message."""
        if self.extra:
            context_str = " | ".join(f"{k}={v}" for k, v in self.extra.items())
            return f"{context_str} | {msg}", kwargs
        return msg, kwargs

def get_trade_logger(trade_id, strategy=None):
    """
    Get a logger with trade context information.
    
    Args:
        trade_id (str): The ID of the trade
        strategy (str, optional): The strategy being used
    
    Returns:
        LoggerAdapter: Logger adapter with trade context
    """
    logger = get_logger("trading")
    extra = {"trade_id": trade_id}
    if strategy:
        extra["strategy"] = strategy
    
    return LoggerAdapter(logger, extra)

def get_performance_logger():
    """
    Get a logger configured specifically for performance monitoring.
    
    Returns:
        logging.Logger: Performance logger
    """
    perf_logger = logging.getLogger("performance")
    
    # Only configure if not already configured
    if not perf_logger.handlers:
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
        
        # Create a timed rotating handler that rotates at midnight
        handler = TimedRotatingFileHandler(
            os.path.join(os.path.dirname(LOG_FILE), "performance.log"),
            when="midnight",
            backupCount=30  # Keep a month of performance logs
        )
        
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            LOG_DATE_FORMAT
        )
        handler.setFormatter(formatter)
        perf_logger.addHandler(handler)
    
    return perf_logger


if __name__ == "__main__":
    # Example usage
    logger = get_logger("test_module")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Example with exception logging
    try:
        x = 1 / 0
    except Exception as e:
        log_exception(logger, e, "Division calculation failed")
    
    # Example with trade logger
    trade_logger = get_trade_logger("T12345", "momentum")
    trade_logger.info("Analyzing potential trade")
    
    # Example with performance logger
    perf_logger = get_performance_logger()
    perf_logger.info("API response time: 235ms")