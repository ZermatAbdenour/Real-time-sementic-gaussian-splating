
import os
import time
import cv2
class DataLoader:
    
    def __init__(self,rgb_path,depth_path = None,stream = False):
        self._rgb_path = rgb_path
        self._depth_path = depth_path

        self.RGBD_pairs = []
        self.time_stamps = []
        if(depth_path is None):
            self.isDepth = False # a variable to indicate if depth is provided
        else:
            self.isDepth = True
        self.current_frame_index = 0
        self.current_keyframe_index = 0
        self._served_last_frame = False
        self.stream = stream
        self._stream_start_time = -1

        #key frame

        #the rgb key framess savess the frames that has translation > alpha and rotation > theta from the last key frames this save training time and performance
        self.rgb_keyframes=[]
        #every time we process and add a batch of frames to the point cloud we clear the depth data to save the performance 
        self.depth_keyframes=[]

    def get_next_frame(self):
        if self._stream_start_time == -1:
            print("Starting Stream...")
            self._stream_start_time = time.time()

        # Non-stream mode: return next frame until finished
        if not self.stream:
            if self.current_frame_index >= len(self.RGBD_pairs):
                return None
            frame = self.RGBD_pairs[self.current_frame_index]
            self.current_frame_index += 1
            return frame

        # Stream mode
        if self.current_frame_index >= len(self.RGBD_pairs):
            return None

        elapsed_time = time.time() - self._stream_start_time
        t0 = self.time_stamps[0]
        next_t = self.time_stamps[self.current_frame_index] - t0

        if elapsed_time < next_t:
            return None

        # Time reached -> serve exactly one frame
        frame = self.RGBD_pairs[self.current_frame_index]
        self.current_frame_index += 1
        return frame

    def __iter__(self):
        return iter(self.RGBD_pairs)
    def __len__(self):
        return len(self.RGBD_pairs)
    
    def load_data(self, limit=-1):
        assert not self.stream, "The default load_data method does not support streaming mode"

        # Load file names in order
        rgb_sorted = sorted(os.listdir(self._rgb_path))
        depth_sorted = sorted(os.listdir(self._depth_path)) if self.isDepth else []

        max_frames = len(rgb_sorted) if limit == -1 else min(limit, len(rgb_sorted))

        for i in range(max_frames):
            print(f"Loading frame {i+1}/{max_frames}", end='\r')

            rgb_file = os.path.join(self._rgb_path, rgb_sorted[i])

            if self.isDepth and i < len(depth_sorted):
                depth_file = os.path.join(self._depth_path, depth_sorted[i])
            else:
                depth_file = None

            # Store paths only
            self.RGBD_pairs.append((rgb_file, depth_file))
