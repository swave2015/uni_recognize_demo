import logging

class Logger:
    def __init__(self, logfile='application.log'):
        # Setting up the root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)  # Log everything
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler for logging
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        
        # Stream handler (console) for logging
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)  # Log info and above to console
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

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


# Example usage:
# log = Logger('my_logfile.log')
# log.debug("This is a debug message")
# log.info("This is an info message")
# log.warning("This is a warning message")
# log.error("This is an error message")
# log.critical("This is a critical message")
