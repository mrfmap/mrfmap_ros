#!/usr/bin/env python
import numpy as np
import os
import glob
import pdb

noises = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

num_samples = 10

main_dir = '/home/icoderaven/bagfiles/'
datasets = ['coffin_world_640','airsim_factory']

for dataset in datasets:
    print 'Sampling random depth images for dataset '+dataset
    for noise in noises:
        directory_path = main_dir + dataset + '/depth_noise_'+str(noise).replace('.', '_')
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    
    # Determine the number of cameras inside this directory
    file_list = glob.glob(main_dir + dataset + '/depth_*.npy')
    cam_list = range(len(file_list))
    for image_index in cam_list:
        print 'Loading image ' + str(image_index)
        depth_img = np.load(main_dir + dataset + '/depth_' + str(image_index) + '.npy')
        pose = np.load(main_dir + dataset + '/pose_' + str(image_index) + '.npy')
        # Inject noise into the depth image
        for noise in noises:
            for j in range(num_samples):
                noisy_depth = depth_img * \
                    (1.0 + noise*np.random.randn(depth_img.shape[0], depth_img.shape[1]))
                directory_path = main_dir + dataset + '/depth_noise_'+str(noise).replace('.', '_')
                dest_path = directory_path+'/'+str(j)+'_depth_'+str(image_index)
                if not os.path.isfile(dest_path):
                    np.save(dest_path, noisy_depth)
                # np.save(directory_path+'/'+str(j)+'_pose_'+str(image_index), pose)