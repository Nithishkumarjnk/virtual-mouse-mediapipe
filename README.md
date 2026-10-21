# Virtual Mouse Using Hand Gestures

A Python computer vision project that allows users to control the mouse cursor using hand gestures captured by a webcam.

## Features
- Real-time hand tracking using MediaPipe Tasks
- Mouse cursor movement using the index fingertip
- Smooth cursor movement
- Left-click using a thumb and index finger pinch gesture
- Custom cursor indicator and status dashboard
- Live webcam preview using OpenCV

## Technologies Used
- Python
- OpenCV
- MediaPipe Tasks
- NumPy
- PyAutoGUI

## Project Setup

1. Install Python.
2. Install the required libraries:

   `pip install -r requirements.txt`

3. Make sure `hand_landmarker.task` is in the project folder.
4. Run the program:

   `python virtual_mouse.py`

## Controls
- Move your index finger to control the mouse cursor.
- Pinch your thumb and index finger to perform a left-click.
- Press `Q` while the webcam window is focused to quit.

## Author
Nithish Kumar J