# MovingObjDetector-WAMI.python

## Use
python WAMI_detector.py -I WAMI-input\ -O WAMI-output\ -N Models\ -M Customised -NT 5

All the images in WAMI-input will be processed. Starating from image 6 (number_of_template + 1). The first 5 frames (NT) are used to build background.

## Dependencies
python 3.7

opencv-python==3.4.2.16

opencv-contrib-python==3.4.2.16

Newest opencv doesn't include SIFT/SURF feature tool, so old version is necessary unless building your opencv-python and the contrib by yourself.

tensorflow==1.14.0

## Reference

https://arxiv.org/abs/1911.01727
