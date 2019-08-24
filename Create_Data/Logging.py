import logging
from datetime import datetime
import os
import sys


class Logger:

    def __init__(self, filename):
        """

        :param filename: String
         The name the log file will have
        """
        self.filename = filename
        self.time = str(datetime.now()).split()[1].split(".")[0].replace(":", "_")
        self.date = str(datetime.now()).split()[0].replace("-", "_")
        self.path = r"C:\Users\Gilad Gecht\PycharmProjects\DepressionResearch\Logs\\"

        self.logger = logging.getLogger(self.path + str(filename) + "_" + self.date + "_" + self.time + ".log")
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s:%(message)s')
        self.file_handler = logging.FileHandler(self.path + str(filename) + "_" + self.date + "_" + self.time + ".log")
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)

    def log(self, message):
        """

        :param message: String
         A message or output which will be written to the log file
        :return:
        """
        self.logger.info(" " + "{}".format(message))
