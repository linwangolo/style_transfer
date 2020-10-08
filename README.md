# Style transfer (Cartoonify)
Cartoonify images with face detection for real-time face style transfer by giving frames from camera.
This repo is based on [stylegan2](https://github.com/justinpinkney/stylegan2) but add face detection before style transfer.
It can identify multiple faces and do multi-face style transfer.
For now, it only support inference for cartoon transfer. If you want to make a different style transfer, using training code in the url above and replace `blended_url` in `StyleTransfer()` with your .pkl.

## Installation

```bash
pip install -r requirements.txt
```


## Pre-trained model

It will automatically download in `StyleTransfer` object when it is created.
The download url can be changed in arguments.


## How to use?

### import images
Using PIL to read images and make it as a list.
```python
from PIL import Image
img = Image.open(PATH)
img_ls = [img1, img2, ...]
```

### Face detection

```python
from image_animation import FaceDetect
detector = FaceDetect()
boxes, probs, annotates, faces = detector.detect(img_ls, crop_size=None, mode = 'Extract_largest', save_faces = True, save_path = 'face_result')
```
If images in list are not in a same size, set `crop_size`.

There're 4 modes: `Detect_bool`, `Detect`, `Extract_largest`, `Extract_all`.

Images of faces will be saved in `save_path = face_result/faces`.
If you want to get the annotation image (face boxes on image), set `save_annotate=True` and it will be save in `face_result/annotations`.

Use `help(FaceDetect)` to see more arguments and details.


### Style transfer
```python
trans = StyleTransfer()
trans.transfer(raw_dir = 'face_result/faces', result_dir = 'style_transfer/results')
```
Suggest to use the face result of face detection as `raw_dir` in case that there're other faces in the original image.

Result image will be save in `result_dir` named `result-toon.jpg`.

## Inference time
The run time from face detection to style transfer is about 89.43s with 1 GPU(GTX 1080), except the very first time needs to download pre-trained model from cloud.

