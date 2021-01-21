# from https://github.com/rolux

import os
import sys
import bz2
from tensorflow.keras.utils import get_file
from style_transfer.ffhq_dataset.face_alignment import image_align
from style_transfer.ffhq_dataset.landmarks_detector import LandmarksDetector

# config
# LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
# RAW_IMAGES_DIR = 'raw'
# ALIGNED_IMAGES_DIR = 'aligned'

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def align(landmark_model, raw_dir, aligned_dir):
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               landmark_model, cache_subdir='temp'))

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in [x for x in os.listdir(raw_dir) if x[0] not in '._']:
        raw_img_path = os.path.join(raw_dir, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join(aligned_dir, face_img_name)
            os.makedirs(aligned_dir, exist_ok=True)
            image_align(raw_img_path, aligned_face_path, face_landmarks)


if __name__ == "__main__":
    landmark_model = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    raw_dir = 'raw'
    aligned_dir = 'aligned'

    align(landmark_model, raw_dir, aligned_dir)
