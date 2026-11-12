# Impenetrable Safe

A safe that opens when you show the right 3-digit code with your fingers under a webcam.
A convolutional neural network (PyTorch, built from scratch) counts the fingers; an Arduino UNO Q
runs the trained model, drives the servo latch, reads the start button and shows the code on its LED matrix.

ME193 Robotics midterm.

## Layout

| Folder | What |
|---|---|
| `vfit/` | Warm-up: fitting the V data from Anna's Code with hand-written gradient descent (single neuron, then 2 ReLU neurons) |
| `camera/` | `fov_check.py`: measure the webcam's field of view with a ruler |
| `lock/` | The finger-counting CNN and the lock |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lock workflow (on the computer)

```bash
python lock/collect.py                 # drag the square over the red square, record 0-5 fingers
python lock/train.py                   # train the CNN -> models/finger_cnn.pt, results/*.png
python lock/lock_live.py --count-only  # check the finger count live
python lock/lock_live.py --set-code 315
python lock/lock_live.py               # SPACE = button
```

## The network

```
3x64x64 image -> conv 5x5 (16) -> ReLU -> maxpool  -> 16x32x32
              -> conv 3x3 (32) -> ReLU -> maxpool  -> 32x16x16
              -> conv 3x3 (64) -> ReLU -> maxpool  -> 64x8x8
              -> flatten (4096) -> linear -> 6 scores (0-5 fingers), cross-entropy loss
```
