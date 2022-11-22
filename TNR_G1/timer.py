import atexit
import warnings
from time import time
from functools import reduce

class Timer():

    start_time = 0

    @staticmethod
    def seconds_to_str(t):
        return "%d:%02d:%02d.%03d" % \
            reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
                [(t*1000,),1000,60,60])

    def end(self):
        print("Program exits, elapsed time:",
              self.seconds_to_str(time() - self.start_time))

    def print_elapsed(self):
        print("Elapsed time:", self.seconds_to_str(time() - self.start_time))

    def print_time(self):
        print("Time now:", self.seconds_to_str(time()))

    def start(self):
        self.start_time
        if self.is_running():
            warnings.warn("Timer can not start because it's already running.")
        else:
            self.start_time = time()
            atexit.register(self.end)

    def stop(self):
        self.start_time
        self.start_time = 0
        atexit.unregister(self.end)

    def is_running(self):
        return self.start_time != 0

