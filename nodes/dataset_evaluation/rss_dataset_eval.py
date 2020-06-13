#!/usr/bin/env python
import numpy as np
from mrfmap_ros.GVDBPyModules import GVDBInference, GVDBImage, gvdb_params, GVDBMapLikelihoodEstimator, KeyframeSelector, GVDBOctomapWrapper, PangolinViewer
from mrfmap_ros.MRFMapRosPyModules import GVDBBatchMapCreator
from mrfmap_ros.ReadTemporalCorrectedBag import ReadTemporalCorrectedBag
from mrfmap_ros.MRFMapGenerator import MRFMapGenerator
import pdb
from geometry import se3, SE3
import click
import yaml
import os
import rospy

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

dtype = np.float32

# Script options!
config_dir = '../../config/'
main_dir = '/home/icoderaven/bagfiles/'
didx = 1    # Dataset ID 0,1, or 2
plot_timings = True
show_map = True # Visualise map in Mayavi. Might have to restart script afterwards because of memory shenanigans
closeup = True  # Show the map in Mayavi from hand picked camera pose
num_repeats = 1 # Multiple repeats, if required for timing data std.dev.


datasets = ['aditya3', 'ICL', 'coffin_world_640', 'cactus_garden']
bagfiles = ['aditya3.bag', 'living_room_with_rgb_noise.bag', 'coffin_world_640.bag', 'cactus_garden.bag']
# Read config files
rs_config_files = [config_dir+'realsense_848.yaml', config_dir+'ICL.yaml',
                   config_dir+'coffin_world_640.yaml', config_dir+'cactus_garden.yaml']
kin_config_files = [config_dir+'kinectone_new.yaml', config_dir+'ICL_gt.yaml',
                    config_dir+'coffin_world_640_gt.yaml', config_dir+'cactus_garden_gt.yaml']

dataset = datasets[didx]
bagfile = bagfiles[didx]
rs_config_file = rs_config_files[didx]
kin_config_file = kin_config_files[didx]

path_str = main_dir + dataset + '/'
bag_file_path = main_dir + dataset + '/' + bagfile

topics_to_parse = []
scale_factors = []
img_lags = []
Ks = []
img_dims = []
bias_poly_files = []
sigma_poly_files = []
cam_in_bodys = []
kf_thresholds = []
use_octos = []
is_icl_bag = False
world_frame_transform = np.eye(4, dtype=dtype)
for files in [rs_config_file, kin_config_file]:
    with open(files) as f:
        node = yaml.load(f)
        viewer_node = node['viewer_params_nodes']
        cam_in_bodys.append(
            np.array(viewer_node['cam_in_body']).reshape(4, 4).astype(dtype))
        topics_to_parse.append(viewer_node['camera_topic'])
        topics_to_parse.append(viewer_node['odom_topic'])
        scale_factors.append(viewer_node['img_scale_factor'])
        img_lags.append(viewer_node['img_lag'])
        kf_thresholds.append(
            [viewer_node['rotation_thresh'], viewer_node['translation_thresh']])
        use_octos.append(viewer_node['view_octomap'])
        is_icl_bag = viewer_node['is_icl_bag']
        if is_icl_bag:
            world_frame_transform = np.array(viewer_node['world_frame_transform']).reshape(4, 4).astype(dtype)
        gvdb_node = node['gvdb_params']
        Ks.append(np.array(gvdb_node['K']).reshape(3, 3).astype(dtype))
        img_dims.append([gvdb_node['rows'], gvdb_node['cols']])
        bias_poly_files.append(gvdb_node['depth_bias_lookup'])
        sigma_poly_files.append(gvdb_node['depth_sigma_lookup'])

# Check if map already exists, if not, do we want to overwrite it?
generate_mrfmap = False
params = gvdb_params()
params.load_from_file(rs_config_file)

suffix = '_res_' + str(dtype(params.res)).replace('.', '_') + '_rot_' + \
    str(dtype(kf_thresholds[0][0])).replace('.', '_') + '_trans_' + str(dtype(kf_thresholds[0][1])).replace('.', '_')

if params.use_polys:
    suffix += 'poly'
# Add special suffix for ICL
if topics_to_parse[0] == '/camera/depth/noisy_image':
    suffix += 'noisy'

if os.path.exists(main_dir+dataset+'/generated/mrfmap' + suffix + '.npy'):
    if raw_input('Overwrite mrfmap? y/n') in ['y', 'Y']:
        generate_mrfmap = True
else:
    generate_mrfmap = True

mrfmap = MRFMapGenerator(
    rs_config_file, main_dir+dataset+'/generated/', 'mrfmap'+suffix)

if generate_mrfmap:
    if plot_timings:
        repeat = num_repeats
    else:
        repeat = 1

    for i in range(repeat):
        creator = GVDBBatchMapCreator(rs_config_file)
        creator.load_bag(bag_file_path)
        creator.process_saved_tuples()

    mrfmap.inference = creator.get_inference_ptr()
    print 'about to save mrfmap'
    mrfmap.save_map()
    print 'done!'
    if show_map:
        mrfmap.show_map(didx)
else:
    if show_map:
        mrfmap_npy = np.load(main_dir+dataset+'/generated/mrfmap'+suffix+'.npy')
        mrf_like_estimator = GVDBMapLikelihoodEstimator(mrfmap.params, False)
        mrf_like_estimator.load_map(mrfmap_npy[:, :3].astype(
            dtype), mrfmap_npy[:, 3].astype(dtype))
        mrfmap.inference = mrf_like_estimator
        mrfmap.load_brick_data()
        mrfmap.show_map(didx, 0.1, True, closeup)

if use_octos[0]:
    if generate_mrfmap:
        print 'Also saving octomap!'
        # Save the generated octomap as well
        mrfmap.inference = creator.get_octomap_ptr()
        mrfmap.title = 'octomap'+suffix
        mrfmap.save_map()
        if show_map:
            mrfmap.show_map(didx)
    else:
        if show_map:
                octomap_npy = np.load(main_dir+dataset+'/generated/octomap'+suffix+'.npy')
                octo_like_estimator = GVDBMapLikelihoodEstimator(mrfmap.params, False)
                octo_like_estimator.load_map(octomap_npy[:, :3].astype(
                    dtype), octomap_npy[:, 3].astype(dtype))
                mrfmap.inference = octo_like_estimator
                mrfmap.title = 'octomap'+suffix
                mrfmap.load_brick_data()
                mrfmap.show_map(didx, 0.5, True, closeup)

raw_input('Done showing maps!!! Press any key to continue')

if generate_mrfmap:
    time_str = creator.get_stats().split('\n')
    types_to_time = ['inference', 'octomap']
    means = []
    stddevs = []
    # Show timings
    if plot_timings:
        timing_fig = plt.figure()
    for ttype in types_to_time:
        # Pull out all the octomap vs mrfmap instances
        entries = [t for t in reversed(time_str) if ttype in t]
        # Pull out mean and std.dev
        means.append(np.array([float(t.split('total=')[-1].split('s')[0]) for t in entries]))
        stddevs.append(np.array([float(t.split('std. dev=')[-1].split('s')[0]) for t in entries]))
        # And save this as well
        np.save(main_dir+dataset+'/generated/' + ttype + suffix + '_timings_means.npy', means)
        np.save(main_dir+dataset+'/generated/' + ttype + suffix + '_timings_stddevs.npy', stddevs)

        if plot_timings:
            # Plot them?
            plt.plot(means[-1], label=ttype)
            plt.fill_between(range(means[-1].shape[0]), means[-1] - 3*stddevs[-1], means[-1] + 3*stddevs[-1], alpha=0.2)
            plt.xlabel('Image index')
            plt.ylabel('Time taken (s)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

    if plot_timings:
        plt.show()

    if generate_mrfmap:
        del creator

# Ok, great. Now remember all those poses and images we had loaded?
# Now create a new set of gvdb params for kinect
print 'Loading params from file!'
# Init gvdb params
old_params = params
params_config_file = kin_config_file
params = gvdb_params()
params.load_from_file(params_config_file)

# Sanity check
if not np.allclose(params.dims, old_params.dims):
    print 'Map dimensions are not the same!!! Fix config file!'
    raise SystemExit
if not np.allclose(params.res, old_params.res):
    print 'Map resolutions are not the same!!! Fix config file!'
    raise SystemExit

# Load map
mrfmap_npy = np.load(main_dir+dataset+'/generated/mrfmap' + suffix + '.npy')
mrf_like_estimator = GVDBMapLikelihoodEstimator(params, False)
mrf_like_estimator.load_map(mrfmap_npy[:, :3].astype(
    dtype), mrfmap_npy[:, 3].astype(dtype))

octomap_npy = np.load(main_dir+dataset+'/generated/octomap' + suffix + '.npy')
octomap_like_estimator = GVDBMapLikelihoodEstimator(params, False)
octomap_like_estimator.load_map(octomap_npy[:, :3].astype(
    dtype), octomap_npy[:, 3].astype(dtype))

estimators = [mrf_like_estimator, octomap_like_estimator]
estimator_labels = ['MRF', 'Octo']

# Read the bag file for kinect data
bag_reader = ReadTemporalCorrectedBag(
    topics_to_parse, bag_file_path, img_lags, scale_factors, world_frame_transform)
bag_reader.read()
print 'Reading done! Computing accuracies...'

accs = []
compute_roc = False

simple_accs = [[], []]
accs_for_simple_accs = np.linspace(0.0, 1.0, 21)

with click.progressbar(range(len(bag_reader.kin_data)),
                       length=len(bag_reader.kin_data)) as bar:
    for index in bar:
        img = bag_reader.kin_data[index][0].astype(dtype)
        pose = bag_reader.kin_data[index][1].astype(dtype)

        pose_kinect = np.dot(pose, cam_in_bodys[1]).astype(dtype)

        acc = np.array([0., 0.])
        for i, estimator in enumerate(estimators):
            likeimg = np.zeros(img.shape, dtype=dtype)
            accimage = np.zeros(img.shape, dtype=dtype)

            # estimator.get_likelihood_image_at_pose_with_depth(pose_kinect, img, likeimg)
            acc[i] = estimator.get_accuracy_image_at_pose_with_depth(pose_kinect, img, accimage)
            if compute_roc:
                simples = []
                gvdb_img = GVDBImage(img)
                for acc_thresh in accs_for_simple_accs:
                    estimator.set_acc_thresh(acc_thresh)
                    accimage = np.zeros(img.shape, dtype=dtype)
                    simples.append(estimator.get_simple_accuracy_at_pose(pose_kinect, gvdb_img))
                simple_accs[i].append(simples)
        accs.append(acc)


accs = np.array(accs)

# Add a little interactivity to see individual likelihood images
plt.close()
acc_fig = plt.figure()
nrows = len(estimators)+1

ax_data = plt.subplot2grid((nrows, 4), (0, 0), colspan=3, picker=True)
ax_depth_img = plt.subplot2grid(
    (nrows, 4), (1, 0), colspan=1, title='Depth')

ax_like_imgs = []
ax_acc_imgs = []
ax_simple_acc_imgs = []

ax_like_imshow_handles = []
ax_acc_imshow_handles = []
ax_simple_acc_imshow_handles = []

test_img = np.random.rand(
    bag_reader.kin_data[0][0].shape[0], bag_reader.kin_data[0][0].shape[1])

for i, estimator in enumerate(estimators):
    ax_like_imgs.append(plt.subplot2grid((nrows, 4), (i+1, 1), colspan=1, title=estimator_labels[i]+' Likelihood'))
    ax_acc_imgs.append(plt.subplot2grid((nrows, 4), (i+1, 2), colspan=1,
                                        title=estimator_labels[i]+' Accuracy', picker=True))
    ax_simple_acc_imgs.append(plt.subplot2grid((nrows, 4), (i+1, 3), colspan=1,
                                               title=estimator_labels[i]+' Simple Accuracy'))

    ax_like_imshow_handles.append(ax_like_imgs[-1].imshow(test_img, interpolation='none', vmin=-7, vmax=7))
    ax_acc_imshow_handles.append(ax_acc_imgs[-1].imshow(test_img, interpolation='none', vmin=-1, vmax=2))
    ax_simple_acc_imshow_handles.append(ax_simple_acc_imgs[-1].imshow(test_img, interpolation='none', vmin=-1, vmax=2))

ax_depth_imshow_handle = ax_depth_img.imshow(
    bag_reader.kin_data[0][0], interpolation='none', vmin=0, vmax=5.0)

ax_data.plot(accs[:, 0], label='MRFMap')
ax_data.plot(accs[:, 1], label='OctoMap')
ax_data.legend()
ax_data.set_title('Accuracies at {0}m'.format(str(dtype(params.res))))
pressed = False
selected_idx = -1

# Add slider for set_acc_thresh
ax_slider = plt.subplot2grid((nrows, 4), (nrows-1, 0), colspan=1)
acc_thresh_slider = Slider(ax_slider, 'set_acc_thresh', 0.0, 1.0, valinit=0.1, valstep=0.01)


def slider_update(val):
    for i, estimator in enumerate(estimators):
        img = bag_reader.kin_data[selected_idx][0].astype(dtype)
        pose = bag_reader.kin_data[selected_idx][1].astype(dtype)

        pose_kinect = np.dot(pose, cam_in_bodys[1]).astype(dtype)
        estimator.set_acc_thresh(val)
        # Also update the simple accuracy image
        simpleaccimage = np.zeros(img.shape, dtype=dtype)
        estimator.get_simple_accuracy_image_at_pose_with_depth(
            pose_kinect, img, simpleaccimage)
        ax_simple_acc_imshow_handles[i].set_data(simpleaccimage)
        acc_fig.canvas.draw()


def draw_at_idx(index):
    global selected_idx
    if index != selected_idx:
        # Set the likelihood image to be the one computed at the corresponding index
        img = bag_reader.kin_data[index][0].astype(dtype)
        pose = bag_reader.kin_data[index][1].astype(dtype)

        pose_kinect = np.dot(pose, cam_in_bodys[1]).astype(dtype)

        for i, estimator in enumerate(estimators):
            accimage = np.zeros(img.shape, dtype=dtype)
            simpleaccimage = np.zeros(img.shape, dtype=dtype)
            likeimage = np.zeros(img.shape, dtype=dtype)

            estimator.get_likelihood_image_at_pose_with_depth(
                pose_kinect, img, likeimage)

            estimator.get_accuracy_image_at_pose_with_depth(
                pose_kinect, img, accimage)

            estimator.get_simple_accuracy_image_at_pose_with_depth(
                pose_kinect, img, simpleaccimage)

            ax_acc_imshow_handles[i].set_data(accimage)
            ax_simple_acc_imshow_handles[i].set_data(simpleaccimage)
            ax_like_imshow_handles[i].set_data(likeimage)
        ax_depth_imshow_handle.set_data(bag_reader.kin_data[index][0])

        # Also display line corresponding to selected index
        for obj_handle in ax_data.findobj(match=type(plt.vlines(0, 0, 0))):
            obj_handle.remove()
        ax_data.vlines(index, 0, 1.0, linestyles='dotted', alpha=0.8)

        acc_fig.canvas.draw()
        selected_idx = index


def onmove(event):
    if pressed and event.inaxes is ax_data:
        draw_at_idx((event.xdata + 0.5).astype(np.int))


def onpress(event):
    global pressed
    if event.button == 1:
        pressed = True
        if event.inaxes is ax_data:
            draw_at_idx((event.xdata + 0.5).astype(np.int))
        if event.inaxes in ax_acc_imgs:
            indx_x = (event.xdata + 0.5).astype(np.int)
            indx_y = (event.ydata + 0.5).astype(np.int)
            estimators[0].set_selected_x_y(indx_x, indx_y)


def onrelease(event):
    global pressed
    if event.button == 1:
        pressed = False
        if event.inaxes is ax_data:
            draw_at_idx((event.xdata + 0.5).astype(np.int))


cid = acc_fig.canvas.mpl_connect('button_press_event', onpress)
rid = acc_fig.canvas.mpl_connect('button_release_event', onrelease)
mid = acc_fig.canvas.mpl_connect('motion_notify_event', onmove)
acc_thresh_slider.on_changed(slider_update)
draw_at_idx(0)
# plt.tight_layout()
plt.savefig(main_dir+dataset+'/generated/octomap' +
            suffix + '_accuracies.pdf', bbox_inches='tight')


plt.show()

np.save(main_dir+dataset+'/generated/octomap' +
        suffix + '_accuracies.npy', accs)

# TODO: Directly print latex table
# from tabulate import tabulate

print np.mean(accs, axis=0)
print np.std(accs, axis=0)

if compute_roc:
    # Also show ROC-ish curve
    fig = plt.figure()
    simple_accs = np.array(simple_accs)
    for i, blah in enumerate(estimator_labels):
        means = np.mean(simple_accs[i], axis=0)
        stddev = np.std(simple_accs[i], axis=0)
        plt.plot(accs_for_simple_accs, means, label=estimator_labels[i])
        plt.fill_between(accs_for_simple_accs, means - stddev, means + stddev, alpha=0.25, linewidth=0.0)
    plt.legend()
    plt.show()
# print 'Done! Dropping to pdb'
# pdb.set_trace()
rospy.signal_shutdown("Done!")
