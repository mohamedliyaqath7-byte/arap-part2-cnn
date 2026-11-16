"""
webots_capture_for_cnn.py  -  ARAP Project Part 2 (Webots controller)

The e-puck stands still, looks at the object you placed in front of it, and
saves one camera picture. The CNN is NOT run inside Webots: deep learning
needs a lot of computing power, so the picture is classified outside the
robot with predict_images.py (as the assignment brief explains).

How to use:
  1. Place an object (for example a box with a picture of a car on it) about
     10 to 20 cm in front of the e-puck.
  2. Set this file as the e-puck controller and run the simulation.
  3. Copy webots_images/webots_capture.png next to predict_images.py
     and run:  python predict_images.py webots_images
"""
from controller import Robot

TIME_STEP = 32
WARM_UP_STEPS = 20          # let the camera settle before taking the picture
SAVE_FOLDER = "webots_images"

import os

robot = Robot()
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

os.makedirs(SAVE_FOLDER, exist_ok=True)
steps = 0
while robot.step(TIME_STEP) != -1:
    steps += 1
    if steps == WARM_UP_STEPS:
        path = os.path.join(SAVE_FOLDER, "webots_capture.png")
        camera.saveImage(path, 100)          # 100 = best quality
        print(f"Saved camera image: {path} ({camera.getWidth()} x {camera.getHeight()} pixels)")
        break
