## bcj_detect.py

### Description

This is a modified version of yolov5's detect.py.  

The un-modified version of detect.py is used to detect garments and their type.

The modifications add these features:
- crop out a central patch of each detected garment
- predict a fabric pattern (ex. plaid, floral, stripe, plain, etc.)
- determine the predominant color

This is implemented by loading a fabric pattern model and using a python library to calculate the color, then translating it to one of 16 human-understandable colors as specified in CSS1.


### Example

An example of the output may be viewed in this video clip:

https://drive.google.com/file/d/1-aoE3e0r4VL_TMoghIlD152nc4IqJfG7/view?usp=sharing


### Installation

bcj_detect.py requires the installation of 2 python libraries:

  pip install fast_colorthief
  pip install webcolors


### Usage

To run bcj_detect.py, execute:

  python3 bcj_detect.py --weights df2.pt --weights-pattern fabric1.pt --source 0 --nosave

    Where:
      --weights: The garment model (assumed to be max 640px by 640px).
      --weights-pattern: The fabric pattern model (assumed to be max 320px by 320px).
      --source 0: The webcam.
      --nosave: No output files are saved.  Removing this creates an .mp4 clip of the entire session.

    Additional optional arguments are described in the yolov5 documentation and in the code.


### Example
