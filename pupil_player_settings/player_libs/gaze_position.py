
class GazePosition:
    def __init__(self,srf=None,aoi=None,timestamps=None):
        self.aoi = aoi
        self.srf = srf
        if timestamps is None:
            self.timestamps = []
        else:
            self.timestamps = timestamps



