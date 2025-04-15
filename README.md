# Lawn_Sensor

This is the Lawn Sensor Capstone Project developed by Team SP25-29. The website visualizes drone data using a modern, modular design. Our drone captures images of various plant species in lawns and, using machine learning, classifies them while measuring plant health, moisture, height, and leaf color. Explore the detailed data table, interactive map and gallery, and a suite of charts for an in-depth view of your lawn's vegetation.

## Dependencies

```bash
pdm add numpy
pdm add pillow
pdm add opencv-python
pdm add scipy
pdm add matplotlib
pdm add ultralytics
```

## Running the Project

```bash
pdm run python python/app.py
```

## Jupyter Notebooks

For running the Jupyter notebooks in the python file, open the notebook, and press "Run All". This may take a few minutes to complete.

## Python Terminal Command

the output of the `generateYoloimagesand` and `generateTGIimage` is a string, which will be in the website (statistic section).

```bash
# generateOutputFrames(video_path, output_dir, fraction)
python python/app.py frames gopro1.mp4

# generateTGIimage(inputString)
# generateYOLOimages(inputString)
python python/app.py analysis 2
```
