import numpy

class observer():
    def __init__(self, location):
        self.location = location
        self.obs_obj = None

    def observed_object(self, obj):
        self.obs_obj = obj
