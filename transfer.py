# import pretrained_networks
from style_transfer import pretrained_networks
from style_transfer import align_images
from style_transfer import project_images
from style_transfer import dnnlib
from style_transfer.dnnlib import tflib

import os
import numpy as np
from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN, extract_face
from PIL import ImageDraw 


class StyleTransfer:
    def __init__(self, network_pl = 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl',
                blended_url="https://drive.google.com/uc?id=1H73TfV5gQ9ot7slSed_l-lim9X7pMRiU"):
        
        self.Gs, self.Gs_blended = self._load_models(network_pl, blended_url)

    def _load_models(self, network_pl, blended_url):
        _, _, Gs = pretrained_networks.load_networks(network_pl)
        _, _, Gs_blended = pretrained_networks.load_networks(blended_url)
        return Gs, Gs_blended

    def transfer(self, raw_dir='style_transfer/raw', processed_dir='style_transfer/aligned',
                 projected_dir='style_transfer/projected', result_dir='style_transfer/results'):
        """

        Args:
            raw_dir (str): Directory of the image to be transferred. Usually use the result directory of face detection or original image.

        """
        landmark_model='http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        align_images.align(landmark_model, raw_dir, processed_dir)
        project_images.project_images(processed_dir, projected_dir, self.Gs,
                   tmp_dir = '.stylegan2-tmp',
                   vgg16_pkl = 'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl', 
                   num_steps = 500, initial_learning_rate = 0.1, initial_noise_factor = 0.05, 
                   verbose=False, video=False)
        
        latent_dir = Path(projected_dir)
        latents = latent_dir.glob("*.npy")
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        for latent_file in latents:
            latent = np.load(latent_file)
            latent = np.expand_dims(latent,axis=0)
            synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
            images = self.Gs_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
            # Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').save(latent_file.parent / (f"{latent_file.stem}-toon.jpg"))
            Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').save(f"{result_dir}/result-toon.jpg")
            
        # delete intermediate files
        os.system(f'rm -r {processed_dir}')
        os.system(f'rm -r {projected_dir}')
        os.system(f'rm -r {raw_dir}')


class FaceDetect:
    def __init__(self, thres = [0.9, 0.9, 0.9], min_face = 100):
        self.mtcnn = MTCNN(thresholds=thres, select_largest=True, post_process=False, device='cuda:0', min_face_size=min_face)
    
    def detect(self, img_ls, crop_size = None, mode = 'Extract_largest', save_faces = True, save_annotate = False, save_path = 'face_result'):
        """face detection

        Args:
            img_ls (list): list of array
            crop_size (tuple, optional): crop images with (left, top, right, bottom). Defaults to None.
            mode (str, optional): There're 3 modes, 'Detect', 'Detect_bool', and 'Extract'. 
                                    If you only want to know whether there're any faces, use 'Detect_bool' mode. 
                                    If you want to get boxes and probs of faces, use 'Detect'.
                                    If you want to get all information about faces, use 'Extract'.
                                    Defaults to 'Detect_bool'.
            face_num (int, optional): Number of faces to be extracted. Defaults to 1.
            save_faces (bool, optional): For 'Extract' mode. Defaults to False.
            save_annotate (bool, optional): For 'Extract' mode. Save images with annotations. Defaults to False.

        Returns:
            tuple: depends on the mode.

        """
        if crop_size:
            for i, img in enumerate(img_ls):
                img_ls[i] = img.crop(crop_size)

        try:
            boxes, probs = self.mtcnn.detect(img_ls)
        except Exception as e:
            print(f'{e} \n...add crop_size=(left, top, right, bottom) to make images the same')

        if mode == 'Detect_bool':
            return isinstance(boxes, np.ndarray)
        elif mode == 'Detect':
            return boxes, probs 
        elif 'Extract' in mode:
            faces = []
            annotates = []
            boxes = boxes.tolist()
            probs = probs.tolist()
            for id_, img in enumerate(img_ls):
                face_batch = []
                img_annotate = img.copy()
                draw = ImageDraw.Draw(img_annotate)
                box_all = boxes[id_]                
                if mode == 'Extract_largest':
                    for i, box in enumerate(box_all):
                        left = max(0, box[0])
                        top = max(0, box[1])
                        right = min(np.array(img_ls[id_]).shape[1], box[2])
                        down = min(np.array(img_ls[id_]).shape[0], box[3])
                        box_all[i] = [left, top, right, down]
                    area = list(map(self._cal_area, box_all))
                    max_id = area.index(max(area))
                    box = box_all[max_id]
                    box_head = [box[0]-box[0]/3, box[1]-box[1]/3, box[2]+box[2]/3, box[3]+box[3]/10]
                    boxes[id_] = [box_head]
                    probs[id_] = [probs[id_][max_id]]

                    draw.rectangle(box_head, width=5)
                    if save_faces:
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        if not os.path.exists(os.path.join(save_path, 'faces')):
                            os.mkdir(os.path.join(save_path, 'faces'))
                        face_batch.append(extract_face(img, box_head, save_path=os.path.join(save_path,'faces', f'detected_face_{id_}-{0}.png')))
                    else:
                        face_batch.append(extract_face(img, box_head))
                elif mode == 'Extract_all':
                    for i, box in enumerate(box_all):
                        box_head = [box[0]-box[0]/8, box[1]-box[1]/5, box[2]+box[2]/8, box[3]+box[3]/10]
                        box_all[i] = box_head
                        draw.rectangle(box_head, width=5)  # box.tolist()
                        if save_faces:
                            if not os.path.exists(save_path):
                                os.mkdir(save_path)
                            if not os.path.exists(os.path.join(save_path, 'faces')):
                                os.mkdir(os.path.join(save_path, 'faces'))
                            face_batch.append(extract_face(img, box_head, save_path=os.path.join(save_path, 'faces', f'detected_face_{id_}-{i}.png')))
                        else:
                            face_batch.append(extract_face(img, box_head))
                else:
                    print(f"Error: there's no mode called {mode}")
                faces.append(face_batch)
                annotates.append(np.asarray(img_annotate))
                if save_annotate:
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    if not os.path.exists(os.path.join(save_path, 'annotations')):
                        os.mkdir(os.path.join(save_path, 'annotations'))
                    img_annotate.save(os.path.join(save_path, 'annotations',f'annotated_faces_{id_}.png'))
            return np.asarray(boxes), np.asarray(probs), annotates, faces
        else:
            print(f"Error: there's no mode called {mode}")

    def _cal_area(self, ls):
        return (ls[2]-ls[0])*(ls[3]-ls[1])
