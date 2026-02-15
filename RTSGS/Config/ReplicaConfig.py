from RTSGS.Config.Config import Config


class ReplicaConfig(Config):
    def __init__(self, config_dict=None):
        super().__init__(config_dict)

        self.set('fx', 320.0)
        self.set('fy', 320.0)
        self.set('cx', 319.5)
        self.set('cy', 239.5)

        self.set('width', 640)
        self.set('height', 480)
        self.set('depth_scale', 1000.0)