from time import gmtime, strftime

class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path

    def log(self, message, with_time=True):
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        with open(self.log_file_path, "a") as f:
            if with_time:
                f.write(f"{time} :  {message}  \n")
            else:
                f.write(f"-----------{message}-----------\n")
