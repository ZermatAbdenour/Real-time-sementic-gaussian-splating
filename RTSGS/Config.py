import numpy as np

class Config:
    def __init__(self, config_dict=None):
        if( config_dict is not None):
            self.config_dict = config_dict
        else:
            self.config_dict = {}
            # set the default value to tum dataset intrinsics
            self.config_dict.setdefault('fx', 525.0)
            self.config_dict.setdefault('fy', 525.0)
            self.config_dict.setdefault('cx', 319.5)
            self.config_dict.setdefault('cy', 239.5)
            self.config_dict.setdefault('width', 640)
            self.config_dict.setdefault('height', 480)
            self.config_dict.setdefault('depth_scale', 5000.0)
            self.config_dict.setdefault('voxel_size', 0.05)
            self.config_dict.setdefault('kf_translation',0.05)
            self.config_dict.setdefault('kf_rotation',5.0 * np.pi / 180.0)

    def get_camera_intrinsics(self):
        intrinsics = np.array([[self.config_dict['fx'], 0, self.config_dict['cx']],
                           [0, self.config_dict['fy'], self.config_dict['cy']],
                           [0, 0, 1]], dtype=np.float32)
        return intrinsics
    def get(self, key, default=None):
        return self.config_dict.get(key, default)

    def set(self, key, value):
        self.config_dict[key] = value

    def to_dict(self):
        return self.config_dict