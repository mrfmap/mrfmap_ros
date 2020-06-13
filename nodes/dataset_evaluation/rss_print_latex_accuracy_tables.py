#!/usr/bin/env python
import numpy as np
from tabulate import tabulate
import pdb

dtype = np.float32

config_dir = '../../config/'
main_dir = '/home/icoderaven/bagfiles/'

datasets = ['aditya3', 'ICL', 'coffin_world_640']
bagfiles = ['aditya3.bag', 'living_room_with_rgb_noise.bag', 'coffin_world_640.bag']
# Read config files
rs_config_files = [config_dir+'realsense_848.yaml', config_dir+'ICL.yaml',
                   config_dir+'coffin_world_640.yaml']
kin_config_files = [config_dir+'kinectone_new.yaml', config_dir+'ICL_gt.yaml',
                    config_dir+'coffin_world_640_gt.yaml']


resolutions = [[0.01, 0.02, 0.05],
               [0.01, 0.02, 0.05],
               [0.01, 0.02, 0.05]]

rot_threshs = [[0.5],
               [1.0, 0.5],
               [1.0]]

trans_threshs = [[0.5],
                 [1.0, 0.5],
                 [1.0]]

use_polys = [[False, False, False],
             [False, False, False],
             [True, True, True]]
for didx in [0,1,2]:  # range(3):
    # Read the accuracies
    dataset = datasets[didx]
    print 'Dataset ' + dataset
    # Suffixes
    r_thr = rot_threshs[didx][0]
    t_thr = trans_threshs[didx][0]

    table_data = {}
    latex_lines = []
    mrf_tuples = []
    octo_tuples = []
    for i, res in enumerate(resolutions[didx]):
        suffix = '_res_' + str(dtype(res)).replace('.', '_') + '_rot_' + str(dtype(r_thr)).replace(
            '.', '_') + '_trans_' + str(dtype(t_thr)).replace('.', '_')

        if use_polys[didx][i]:
            suffix += 'poly'
        # Add special suffix for ICL
        if didx == datasets.index('ICL'):
            suffix += 'noisy'

        accs = np.load(main_dir+dataset+'/generated/octomap' + suffix + '_accuracies.npy')
        mrf_mean, octo_mean = np.mean(accs, axis=0)
        mrf_std, octo_std = np.std(accs, axis=0)
        mrf_tuples.append([mrf_mean, mrf_std])
        octo_tuples.append([octo_mean, octo_std])
    
    tuples = [mrf_tuples, octo_tuples]
    labels = ["MRFMap", "OctoMap"]
    for i,tup in enumerate(tuples):
        num_res = len(resolutions[didx])
        latex_line = []
        latex_line.append(labels[i])
        for j in range(num_res):
            latex_line.append( r"\( {:.3f} \pm {:.3f} \)".format(tup[j][0], tup[j][1]))
            
        latex_lines.append(latex_line)
    
    labels = []
    labels.append("Resolutions")
    colalign = ["left"]
    for res in resolutions[didx]:
        labels.append(str(dtype(res)) + "m")
        colalign.append("center")

    print(tabulate(latex_lines, headers=labels, colalign=colalign, tablefmt="latex_raw"))
