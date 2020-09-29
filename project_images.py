# from https://github.com/rolux/stylegan2encoder

import argparse
import os
import shutil
import numpy as np

from style_transfer import dnnlib
from style_transfer.dnnlib import tflib as tflib
from style_transfer import pretrained_networks
from style_transfer import projector
from style_transfer import dataset_tool
from style_transfer.training import dataset
from style_transfer.training import misc


def project_image(proj, src_file, dst_dir, tmp_dir, video=False):

    data_dir = '%s/dataset' % tmp_dir
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    image_dir = '%s/images' % data_dir
    tfrecord_dir = '%s/tfrecords' % data_dir
    os.makedirs(image_dir, exist_ok=True)
    shutil.copy(src_file, image_dir + '/')
    dataset_tool.create_from_images_raw(tfrecord_dir, image_dir, shuffle=0)
    dataset_obj = dataset.load_dataset(
        data_dir=data_dir, tfrecord_dir='tfrecords',
        max_label_size=0, repeat=False, shuffle_mb=0
    )

    print('Projecting image "%s"...' % os.path.basename(src_file))
    images, _labels = dataset_obj.get_minibatch_np(1)
    images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
    proj.start(images)
    if video:
        video_dir = '%s/video' % tmp_dir
        os.makedirs(video_dir, exist_ok=True)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if video:
            filename = '%s/%08d.png' % (video_dir, proj.get_cur_step())
            misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)

    os.makedirs(dst_dir, exist_ok=True)
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.png')
    misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.npy')
    np.save(filename, proj.get_dlatents()[0])


def project_images(src_dir, dst_dir, Gs,
                   tmp_dir = '.stylegan2-tmp',
                   vgg16_pkl = 'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl', 
                   num_steps = 500, initial_learning_rate = 0.1, initial_noise_factor = 0.05, 
                   verbose=False, video=False):
    # print('Loading networks from "%s"...' % network_pkl)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector(
        vgg16_pkl             = vgg16_pkl,
        num_steps             = num_steps,
        initial_learning_rate = initial_learning_rate,
        initial_noise_factor  = initial_noise_factor,
        verbose               = verbose
    )
    proj.set_network(Gs)

    src_files = sorted([os.path.join(src_dir, f) for f in os.listdir(src_dir) if f[0] not in '._'])
    for src_file in src_files:
        project_image(proj, src_file, dst_dir, tmp_dir, video=video)
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    src_dir = 'aligned'
    dst_dir = 'generated'

    network_pkl = 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl'
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    project_images(src_dir, dst_dir, Gs)
