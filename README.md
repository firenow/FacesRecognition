## Synopsis

This program use the actual user webcam to find all the faces present.
Foreach new face found, it tries to identifies and follows it while it shows.

## Motivation

POC for facial recognition and following.

## Installation

1. Install OpenCV3.1 and have all it's .bin on your $PATH
2. Launch ocvfacerecogn with Python 3.5

## API Reference

OpenCV3.1 : http://opencv.org/opencv-3-1.html

## Tests

N/A

## Known issues

- The measurements are saved as the center position while the predictions are saved as the top-left
- LOTS of various multiprocessing issues

## Contributors

Thierry BARTHEL

## License

MIT License (MIT)