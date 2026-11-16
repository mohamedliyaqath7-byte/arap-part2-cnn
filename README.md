# ARAP Part 2: CNN object recognition for an e-puck in Webots

Applied Robotics and AI Project, University of Hertfordshire (Assignments 3, 4 and 5).
Student: Mohamed Rafi Mohamed Liyaqath (596FYWQK)

A CNN is trained on CIFAR-10 in Google Colab. The e-puck in Webots saves one camera
picture, and a separate Python script classifies it. The CNN runs **outside** Webots,
because deep learning needs more computing power than a robot control loop can spare.

## Results

| Output | Result |
|---|---|
| CNN layers | 30 layers, 551,914 parameters |
| Test accuracy | 85.52% on 10,000 CIFAR-10 test images |
| Own images | GT-R: automobile 100.0%, fighter jet: airplane 99.2%, eagle: airplane 84.9% (wrong), shark: airplane 97.4% (no such class) |
| Webots camera picture | automobile 100.0% |
| Quantization (TFLite) | 556 KB, about 4x smaller than the float32 weights, accuracy 85.4% on 2,000 test images |

## Folder structure

```
cifar10_cnn.py                 build, train and evaluate the CNN (run in Colab)
predict_images.py              classify the pictures in a folder (top 3 guesses)
webots/controllers/webots_capture_for_cnn/   e-puck controller that saves one camera picture
webots/worlds/cnn_test.wbt     Webots world with the e-puck and the car panel
webots/worlds/textures/        put car.jpg here (photo used on the panel)
results/                       metrics from the training run
```

## How to run

1. **Train.** Open Google Colab (GPU), upload `cifar10_cnn.py`, run `!python cifar10_cnn.py`.
   It saves the model, the layer list, `metrics.json` and the graphs in `outputs/`.
2. **Own images.** Put 3 or more pictures in a folder, then run `!python predict_images.py my_images`.
3. **Webots.** Put `car.jpg` in `webots/worlds/textures/`, open `cnn_test.wbt` in Webots R2023b
   and run the simulation. The e-puck saves `webots_images/webots_capture.png`.
4. **Classify the Webots picture.** Copy `webots_capture.png` to the Colab folder `webots_images`
   and run `!python predict_images.py webots_images`.

## Workflow

Work was planned on Trello (lists To Do, Doing with WIP limit 2, Completed) and saved on GitHub
with small commits.

## Hardware and software notes

- e-puck camera set to 256 x 256 pixels (default is 52 x 39).
- Webots R2023b, Python controller, 32 ms time step; the controller waits 20 steps before saving.
- TensorFlow/Keras, TensorFlow Lite, Pillow.

## Limitations

The Webots test uses one picture of a car, so it shows the pipeline works but not that the model
generalises. The timings were measured on Colab, not on an e-puck.
