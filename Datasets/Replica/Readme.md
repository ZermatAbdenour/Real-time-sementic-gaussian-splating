# RGB-D Capture for Replica Scenes

This directory provides scripts to capture RGB-D data from Replica scenes using Habitat-Sim. The captured data can be used with the TumDataLoader by providing the generated trajectory and RGB-D images.

## Quick Start

### Create the Conda environment
Run the following script to set up the required environment:
```bash
./create-conda.sh
```

### Capture RGB-D data
Run the capture script:
```bash
python HabitatSimCapture.py
```

This will generate:
- A `trajectory.txt` file with the camera path
- RGB-D images captured along the trajectory

## Use with TumDataLoader
You can load the captured data by passing the `trajectory.txt` and the habitat_capture folder containing RGB-D images to the TumDataLoader.

## Notes
The RGB-D data captured via Habitat-Sim does not fully reproduce the visual quality of Replica scenes, such as reflections and fine lighting effects. For high-quality RGB-D data, you can either:
- Render the scene yourself using a high-quality renderer, or
- Use the pre-rendered data provided by Nice-SLAM for Replica scenes: [Nice-SLAM Replica Data](https://cvg-data.inf.ethz.ch/nice-slam/data/)