from abc import ABC, abstractmethod

class Tracker(ABC):
    @abstractmethod
    def __init__(self):
        self.poses = []
        self.keyframes_poses=[]

    @abstractmethod
    def track_frame(self, rgb,depth = None):
        pass
    
    @abstractmethod
    def visualize_tracking(self):
        pass