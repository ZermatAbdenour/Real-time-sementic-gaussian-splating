import os
from RTSGS.DataLoader.TUMDataLoader import TUMDataLoader
from RTSGS.System import RTSGSSystem
from RTSGS.Tracker.SimpleORBTracker import SimpleORBTracker
from RTSGS.Config import Config
#from RTSGS.Tracker.SimpleOpen3DVO import SimpleOpen3DVO 

if __name__ == "__main__":

    # Load Data
    print("Starting RTSGS System...")
    data_path = "./data/rgbd_dataset_freiburg3_long_office_household/"
    
    data = TUMDataLoader(os.path.join(data_path, "rgb"), os.path.join(data_path, "depth"),os.path.join(data_path, "groundtruth.txt"), stream=True)
    config = Config()
    print("Loading Data...")
    data.load_data(1500)
    print("Data Loaded.")

    # Initialize Tracker
    tracker = SimpleORBTracker(dataset=data,config=config)

    # Initialize System
    system = RTSGSSystem(data,tracker,config, stream=True)

    system.run()
