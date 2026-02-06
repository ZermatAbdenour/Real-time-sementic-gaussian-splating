
import threading
from RTSGS.GUI.WindowManager import WindowManager
from RTSGS.GaussianSplatting.PointCloud import PointCloud

class RTSGSSystem:
    def __init__(self, dataset, tracker,config, stream=False):
        self.dataset = dataset
        self.tracker = tracker
        self.stream = stream


        self._stop = False
        self._busy = False
        self._pending = None

        self._cv = threading.Condition()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

        # Define the point cloud
        self.pcd = PointCloud(config)

        self.window = WindowManager(self.pcd,1280, 720, "RTSGS System")

        #we combine each Frame point cloud tensor each n frames to save performance
        self.update_point_cloud_each = 50
        self.final_update = False

    def run(self):
        if not self.stream:
            return

        self._worker.start()

        while not self.window.window_should_close():
            self.window.start_frame()

            frame = self.process_stream_frame()
            if frame is not None:
                img, depth = frame[0], frame[1]
                self.pcd.add_frame(img, depth, self.tracker.poses[-1])
            
            #main thread

            #we update the pcd each n frames 
            if(self.dataset.current_frame_index %self.update_point_cloud_each == 0 and self.dataset.current_frame_index< len(self.dataset.RGBD_pairs) and frame is not None ):
                self.pcd.update_full_pointcloud()
            if(self.dataset.current_frame_index== len(self.dataset.RGBD_pairs) and self.final_update == False):
                self.final_update = True
                self.pcd.update_full_pointcloud()

            self.tracker.visualize_tracking()
            self.window.render_frame()

        with self._cv:
            self._stop = True
            self._cv.notify_all()

        self._worker.join(timeout=1.0)
        self.window.shutdown()

    def process_stream_frame(self):
        # Check busy without grabbing frames/decoding when we will skip anyway
        with self._cv:
            if self._busy:
                return None

        frame = self.dataset.get_next_frame()
        if frame is None:
            return None
        img, depth = frame[0], frame[1]
        with self._cv:
            self._pending = (img, depth)
            self._busy = True
            self._cv.notify()
        return frame

    def _worker_loop(self):
        while True:
            with self._cv:
                while not self._stop and self._pending is None:
                    self._cv.wait()

                if self._stop:
                    return

                img, depth = self._pending
                self._pending = None

            try:
                self.tracker.track_frame(img, depth)
            finally:
                with self._cv:
                    self._busy = False