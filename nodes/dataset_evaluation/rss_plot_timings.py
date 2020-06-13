#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mrfmap_ros.GVDBPyModules import gvdb_params
import yaml
import pdb
plt.style.use('seaborn-muted')
plt.rcParams.update({'font.size': 12})
dtype = np.float32

config_dir = '../../config/'
main_dir = '/home/icoderaven/bagfiles/'
didx = 1

resolutions = [0.01, 0.02, 0.05]
rot_thresh = 1.0
trans_thresh = 1.0

datasets = ['aditya3', 'ICL', 'coffin_world_640']
bagfiles = ['aditya3.bag', 'living_room_with_rgb_noise.bag',
            'coffin_world_640.bag']
# Read config files
rs_config_files = [config_dir+'realsense_848.yaml',
                   config_dir+'ICL.yaml', config_dir+'coffin_world_640.yaml']
kin_config_files = [config_dir+'kinectone_new.yaml',
                    config_dir+'ICL_gt.yaml', config_dir+'coffin_world_640.yaml']

dataset = datasets[didx]
bagfile = bagfiles[didx]
rs_config_file = rs_config_files[didx]
kin_config_file = kin_config_files[didx]

path_str = main_dir + dataset + '/'
bag_file_path = main_dir + dataset + '/' + bagfile

types_to_time = ['inference', 'octomap']
labels_for_fig = ['MRFMap', 'OctoMap']
labels_for_title = ['Real-World', 'ICL livingroom1', 'Simulation']

timing_fig, ax = plt.subplots(1, 1, dpi=200)
plt.title(labels_for_title[didx])

params = gvdb_params()
params.load_from_file(rs_config_file)

topics_to_parse = []
for files in [rs_config_file, kin_config_file]:
    with open(files) as f:
        node = yaml.load(f)
        viewer_node = node['viewer_params_nodes']
        topics_to_parse.append(viewer_node['camera_topic'])
        topics_to_parse.append(viewer_node['odom_topic'])

linestyles = ['-', '--']
for res in resolutions:
    color = next(ax._get_lines.prop_cycler)['color']

    for ttype in types_to_time:
        suffix = '_res_' + str(dtype(res)).replace('.', '_') + '_rot_' + \
            str(dtype(rot_thresh)).replace('.', '_') + '_trans_' + \
            str(dtype(trans_thresh)).replace('.', '_')
        if params.use_polys:
            suffix += 'poly'
        # Add special suffix for ICL
        if topics_to_parse[0] == '/camera/depth/noisy_image':
            suffix += 'noisy'

        # Pull out mean and std.dev
        means = np.load(main_dir+dataset+'/generated/' +
                        ttype + suffix + '_timings_means.npy')
        stddevs = np.load(main_dir+dataset+'/generated/' +
                          ttype + suffix + '_timings_stddevs.npy')
        # Plot them?
        label = labels_for_fig[types_to_time.index(ttype)] + ' ' + str(dtype(res)) + 'm'
        ax.plot(means[-1], label=label, color=color, linestyle=linestyles[types_to_time.index(ttype)])
        ax.fill_between(range(means[-1].shape[0]), means[-1] -
                        3*stddevs[-1], means[-1] + 3*stddevs[-1], alpha=0.2)
        ax.set_xlabel('Image index')
        ax.set_ylabel('Time taken (s)')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

plt.savefig(main_dir+dataset+'/generated/' + 'timings.pdf')
plt.show()
