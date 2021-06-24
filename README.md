# PiVA
It's a project to do simple video analytic functions on raspberry pi. It could run on all kinds of platform (x86, arm) theoretically. However, the prebuilt ncnn library is based on raspberry pi model 3b+, which uses an A53 cpu. You're encouraged to build ncnn library based on your architecture. Raspberry pi model above 3b is recommended(3b, 3b+, 4b), Jetson nano should also works really well.

## Prerequisite
opencv
cmake

## Compile
```
./build.sh
```

## How to Use
1. use cropface to crop faces from images and put them in facedb/
2. cd build/install/
3. use makefrdb facedb to construct your own database
4. ./piva 0
5. use control to open/close functions

## Usage
### piva
description: Main loop of PiVA
usage: piva [param]
param: camera dev number. usually 0

### cropface
dedscription: Simple helper function to make face database
usage cropface [param]
param: image path to crop faces
output: faces in image.jpg

### control
description: Control of piva
usage: control [param1] [param2]
param1: fd/fr/bbox/landmark/realface
param2: open/close
example:
```
./control fd open #enable face detection
./control fd close #disable face detection
./control fr open #enable face recognition
./control fr close #disable face recognition
./control bbox open #show face bounding box
./control bbox close #don't show face bounding box
./control landmark #show face landmark
./control landmark #don't show face landmark
```
## Reference
These are all great repositories. Some of the codes and models come from there. Please STAR them!
1. https://github.com/Tencent/ncnn
2. https://github.com/ShiqiYu/libfacedetection
3. https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi
4. https://github.com/MirrorYuChen/ncnn_example

