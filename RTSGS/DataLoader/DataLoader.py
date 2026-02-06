
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
        self._served_last_frame = False
        self.stream = stream
        self._stream_start_time = -1

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
    def load_data(self,limit = -1):
        assert not self.stream, "The default load_data method does not support streaming mode"
        #This method loads the data by name order from the provided paths
        rgb_sorted = os.listdir(self._rgb_path)
        depth_sorted = os.listdir(self._depth_path)
        for i in range(min(limit,len(rgb_sorted))):
            print(f"Loading frame {i+1}/{min(limit,len(rgb_sorted))}",end = '\r')
            rgb_file = os.path.join(self._rgb_path,rgb_sorted[i])
            rgb_image = cv2.imread(rgb_file,cv2.IMREAD_COLOR)
            if(self.isDepth and i < len(depth_sorted)):
                depth_file = os.path.join(self._depth_path,depth_sorted[i])
                depth_image = cv2.imread(depth_file,cv2.IMREAD_UNCHANGED)
            else:
                depth_image = None
        
            self.RGBD_pairs.append((rgb_image,depth_image))