import os

class SkyLog:
    def __init__(self, log_file='skylog.txt'):
        self.log_file = log_file
        self.ensure_log_file()

    def ensure_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write('SkyLog Initialized\n')

    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f'{message}\n')

    def info(self, message):
        self.log(f'INFO: {message}')

    def error(self, message):
        self.log(f'ERROR: {message}')
    
    def warning(self, message):
        self.log(f'WARNING: {message}')

    def read_logs(self):
        with open(self.log_file, 'r') as f:
            return f.readlines()