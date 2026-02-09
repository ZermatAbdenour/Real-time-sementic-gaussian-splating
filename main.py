import os
from RTSGS.DataLoader.TUMDataLoader import TUMDataLoader
from RTSGS.System import RTSGSSystem
from RTSGS.Tracker.SimpleORBTracker import SimpleORBTracker
from RTSGS.Config import Config
#from RTSGS.Tracker.SimpleOpen3DVO import SimpleOpen3DVO 

if __name__ == "__main__":

    # Load Data
    print("Starting RTSGS System...")
    data_path = "./Datasets/Replica/habitat_capture"
    trajectory_path = "./Datasets/Replica/trajectory.txt"
    
    data = TUMDataLoader(os.path.join(data_path, "rgb"), os.path.join(data_path, "depth"),trajectory_path, stream=True)
    config = Config()
    config.set('fx', 320.0)
    config.set('fy', 320.0)
    config.set('cx', 319.5)
    config.set('cy', 239.5)

    config.set('width', 640)
    config.set('height', 480)
    config.set("depth_scale",1000)
    print("Loading Data...")
    data.load_data(1500)
    print("Data Loaded.")

    # Initialize Tracker
    tracker = SimpleORBTracker(dataset=data,config=config)

    # Initialize System
    system = RTSGSSystem(data,tracker,config, stream=True)

    system.run()
