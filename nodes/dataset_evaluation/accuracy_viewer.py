#!/usr/bin/env python
import yaml
import numpy as np
from mrfmap_ros.GVDBPyModules import GVDBInference, GVDBImage, gvdb_params, GVDBMapLikelihoodEstimator, KeyframeSelector, GVDBOctomapWrapper
from mrfmap_ros.MRFMapGenerator import MRFMapGenerator
import cv2
import glob
import pdb
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
plt.style.use('seaborn-muted')
plt.rcParams.update({'font.size': 12})


dtype = np.float32

main_dir = '/home/icoderaven/bagfiles/'
config_dir = '../../config/'
datasets = ['coffin_world_640']
dataset = datasets[0]
path_str = main_dir + dataset + '/'
img_dims = np.array([640, 480])
occ_thresh = 0.5

load_map = False
use_noise = False
use_realsense_noise = True
show_utility = True

K = np.eye(3)
with open(config_dir+'coffin_world_640.yaml') as f:
    K = np.array(yaml.load(f)['gvdb_params']['K']).reshape(3,3)


if use_noise:
    noise_val = 0.0
    noise_str = str(noise_val).replace('.', '_')

if use_realsense_noise:
    # Load the polynomials
    with open(config_dir+'polys/realsense_bias_poly5x.yml') as f:
        bias_poly = np.array(yaml.load(f)).reshape(
            img_dims[1]/20, img_dims[0]/20, 3)
        # Now just repeat to img dims for easy noisy image generation
        bias_poly = np.repeat(np.repeat(bias_poly, 20, axis=0), 20, axis=1)
    with open(config_dir+'polys/realsense_stddev_poly5x.yml') as f:
        stddev_poly = np.array(yaml.load(f)).reshape(
            img_dims[1]/20, img_dims[0]/20, 3)
        # Now just repeat to img dims for easy noisy image generation
        stddev_poly = np.repeat(
            np.repeat(stddev_poly, 20, axis=0), 20, axis=1)

cam_id = 0

# Determine the number of cameras inside this directory
if use_noise:
    file_list = glob.glob(path_str + 'depth_noise_' +
                          noise_str + '/0_depth_*.npy')
else:
    file_list = glob.glob(path_str + 'depth_*.npy')
num_images = len(file_list)
print 'Detected {0} images'.format(num_images)


# Init gvdb params
params_config_file = config_dir+'coffin_world_640.yaml'
params = gvdb_params()
params.load_from_file(params_config_file)
print 'Setting from python...'
params.set_from_python()


# First, read all the images
print 'Reading all the images...'
poses = []
depth_imgs = []
gt_depth_imgs = []
gt_with_bias_imgs = []
image_ptrs = []
list_of_images = range(num_images)
keyframe_selector = KeyframeSelector(0.05, 0.05)

# First, load all the images
num_images = 0
for image_index in list_of_images:
    pose = np.load(path_str+'pose_' + str(image_index) + '.npy')
    cam_in_body = np.eye(4)
    cam_in_body[:3, :3] = np.array(
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    pose_cpp = np.dot(pose, cam_in_body)

    if use_noise:
        depth_img = np.load(path_str + 'depth_noise_' + noise_str +
                            '/0_depth_' + str(image_index) + '.npy')
    else:
        depth_img = np.load(path_str + 'depth_' +
                            str(image_index) + '.npy')
    depth_img_cpp = np.array(depth_img)

    if use_realsense_noise:
        gt_depth_imgs.append(depth_img_cpp)
        # Yay! First apply the bias
        biased_depth_img = depth_img_cpp + \
            bias_poly[:, :, 2] * depth_img_cpp**2 + \
            bias_poly[:, :, 1]*depth_img_cpp + bias_poly[:, :, 0]

        # Save these for computation of the accuracy later...
        gt_with_bias_imgs.append(biased_depth_img)
        # Now get stddev values
        stddev_depth_img = stddev_poly[:, :, 2] * depth_img_cpp**2 + \
            stddev_poly[:, :, 1]*depth_img_cpp + stddev_poly[:, :, 0]
        # Now sample from stddev values and add to biased depth image
        generated_img = biased_depth_img + stddev_depth_img * \
            np.random.randn(img_dims[1], img_dims[0])
        # pdb.set_trace()
        # Assign this to the depth_img_cpp
        depth_img_cpp = generated_img.astype(dtype)

    if keyframe_selector.is_keyframe(pose_cpp.astype(dtype)):
        poses.append(pose_cpp)
        depth_imgs.append(depth_img_cpp)
        num_images += 1

# # write bagfile
# import rosbag

# We need to do cross-validation. So, let's hold out one image, and train on the rest
accs = []
for cam_id in range(num_images):
    if load_map:
        print 'Loading occupancy map...'
        if use_noise:
            title = '_res_' + \
                str(np.float32(resolution)).replace('.', '_')+'_p_' + \
                str(np.float32(prior)).replace('.', '_') + '_n_' + \
                str(np.float32(noise_val)).replace('.', '_')
        elif use_realsense_noise:
            title = '_rs_res_'+str(np.float32(params.res)
                                   ).replace('.', '_') + '_fold_'+str(cam_id)

        directory_path = main_dir+dataset+'/generated/'
        octomap = np.load(directory_path + 'oct' + title + '.npy')
        mrfmap = np.load(directory_path + 'mrf' + title + '.npy')

        print 'Loading maps '+title
        octo_like_estimator = GVDBMapLikelihoodEstimator(False)
        mrf_like_estimator = GVDBMapLikelihoodEstimator(False)
        octo_like_estimator.load_map(octomap[:, :3].astype(
            dtype), octomap[:, 3].astype(dtype))
        mrf_like_estimator.load_map(mrfmap[:, :3].astype(
            dtype), mrfmap[:, 3].astype(dtype))
        estimators = [octo_like_estimator, mrf_like_estimator]
    else:
        # We have to generate our own maps.
        print 'Creating Likelihood object...'
        mrfmap = MRFMapGenerator(params_config_file, main_dir+dataset +
                                 '/generated/', 'test_mrfmap')
        octomap = GVDBOctomapWrapper(params)

        image_ptrs = []

        for image_index in range(num_images):
            if image_index == cam_id:
                # This is the image index we're going to hold out. Select all the other indices
                # Store the ground truth image here for comparison...
                # Need to create a GVDBImage here (due to context issues?)
                pass
                # octo_like_estimator.add_camera(poses[image_index].astype(dtype), gt_with_bias_image_ptrs[-1])
                # mrf_like_estimator.add_camera(poses[image_index].astype(dtype), gt_with_bias_image_ptrs[-1])
            else:
                # print '.Adding image '+str(image_index)
                image_ptrs.append(
                    GVDBImage(depth_imgs[image_index].astype(dtype)))
                # print '..Adding to mrfmap'
                mrfmap.add_data(poses[image_index].astype(
                    dtype), image_ptrs[-1])
                # print '..Adding to octomap'
                octomap.add_camera(
                    poses[image_index].astype(dtype), image_ptrs[-1])

        # Store the corresponding map informations in the likelihood objects...
        octomap.push_to_gvdb_volume()
        estimators = [octomap, mrfmap.inference]

    # Ok, time to get the accuracy images
    gt_with_bias_image_ptrs = []
    gt_with_bias_image_ptrs.append(
        GVDBImage(gt_with_bias_imgs[cam_id].astype(dtype)))
    octoimage = np.zeros(depth_imgs[0].shape, dtype=dtype)
    mrfimage = np.zeros(depth_imgs[0].shape, dtype=dtype)
    acc_o = estimators[0].get_accuracy_image_at_pose(poses[cam_id].astype(
        dtype), gt_with_bias_image_ptrs[-1], octoimage)
    acc_m = estimators[1].get_accuracy_image_at_pose(poses[cam_id].astype(
        dtype), gt_with_bias_image_ptrs[-1], mrfimage)

    if show_utility:
        fig = plt.figure()
        ax_octo = plt.subplot2grid((1, 2), (0, 0), colspan=1, picker=True)
        ax_mrf = plt.subplot2grid((1, 2), (0, 1), colspan=1, picker=True)
        ax_octo.imshow(octoimage)
        ax_octo.set_title('Octomap Accuracy Image')
        ax_mrf.imshow(mrfimage)
        ax_mrf.set_title('MRFMap Accuracy Image')
        plt.show()

    octounique, octo_counts = np.unique(octoimage, return_counts=True)
    mrfunique, mrf_counts = np.unique(mrfimage, return_counts=True)

    accuracy = [float(counts[2])/float(counts[1]+counts[2]+0.01)
                for counts in [octo_counts, mrf_counts]]
    print 'Accuracy is '
    print accuracy
    print acc_o, acc_m
    accs.append([acc_m, acc_o])

    if show_utility:

        octoimage = np.zeros(depth_imgs[0].shape, dtype=dtype)
        mrfimage = np.zeros(depth_imgs[0].shape, dtype=dtype)
        estimators[0].get_accuracy_image_at_pose(poses[cam_id].astype(
            dtype), gt_with_bias_image_ptrs[-1], octoimage)
        estimators[1].get_accuracy_image_at_pose(poses[cam_id].astype(
            dtype), gt_with_bias_image_ptrs[-1], mrfimage)

        fig = plt.figure(dpi=200)
        ax_octo = plt.subplot2grid((2, 2), (0, 0), colspan=1, picker=True)
        ax_octo.set_title('Octomap Accuracy')
        ax_mrf = plt.subplot2grid((2, 2), (0, 1), colspan=1, picker=True)
        ax_mrf.set_title('MRFMap Accuracy')
        ax_data_octo = plt.subplot2grid((2, 2), (1, 0), colspan=1)
        ax_data_mrf = plt.subplot2grid((2, 2), (1, 1), colspan=1)
        ax_imgs = [ax_octo, ax_mrf]
        ax_datas = [ax_data_octo, ax_data_mrf]

        ax_octo_imshow_handle = ax_octo.imshow(octoimage)
        ax_mrf_imshow_handle = ax_mrf.imshow(mrfimage)

        def onclick(event):
            indx_x = (event.xdata + 0.5).astype(np.int)
            indx_y = (event.ydata + 0.5).astype(np.int)
            if event.inaxes is ax_octo or event.inaxes is ax_mrf:
                if event.button == 1:
                    for i, estimator in enumerate(estimators):
                        # Set the x and y
                        estimator.set_selected_x_y(indx_x, indx_y)
                        # Call the diagnostic method
                        diag_image = np.zeros(
                            depth_imgs[0].shape, dtype=dtype).flatten()

                        estimator.get_diagnostic_image_at_pose(poses[cam_id].astype(
                            dtype), gt_with_bias_image_ptrs[-1], diag_image)
                        # Extract data from image
                        # Get num_voxels from last entry
                        stride = 10
                        num_voxels = diag_image[-1].astype(np.int)
                        z_distances = diag_image[:stride*num_voxels:stride]
                        alphas = diag_image[1:stride*num_voxels:stride]
                        w_is = diag_image[4:stride*num_voxels:stride]
                        vis_is = diag_image[5:stride*num_voxels:stride]
                        voxel_ids = np.frombuffer(diag_image, dtype=np.uint64)[
                            1:num_voxels*(stride/2):(stride/2)]
                        node_ids = np.frombuffer(diag_image, dtype=np.uint64)[
                            3:num_voxels*(stride/2):(stride/2)]
                        brick_ids_x = np.frombuffer(diag_image, dtype=np.uint8)[
                            8*4:stride*4*num_voxels:stride*4]
                        brick_ids_y = np.frombuffer(diag_image, dtype=np.uint8)[
                            8*4+1:stride*4*num_voxels:stride*4]
                        brick_ids_z = np.frombuffer(diag_image, dtype=np.uint8)[
                            8*4+2:stride*4*num_voxels:stride*4]

                        brick_ids = np.vstack(
                            (brick_ids_x, brick_ids_y, brick_ids_z))

                        # Plot corresponding data
                        ax_datas[i].clear()
                        ax_datas[i].plot(z_distances, w_is,
                                         label='$\omega$', marker='.', lw=2)

                        ax_datas[i].plot(z_distances, alphas,
                                         label='$p_{occ}$', marker='.', ms=0.5)
                        ax_datas[i].plot(z_distances, vis_is,
                                         label='$vis$', ls='-', lw=1)

                        ax_datas[i].set_ylim(top=1.5, bottom=0.0)
                        print 'Sum of w_is is '+str(np.sum(w_is))

                        d_meas = depth_imgs[cam_id][indx_y, indx_x]
                        print 'Actual depth here is '+str(d_meas)

                        xlims = ax_datas[i].get_xlim()
                        ylims = ax_datas[i].get_ylim()

                        if np.isnan(d_meas):
                            d_meas = 10.0
                        gt_depth = gt_depth_imgs[cam_id][indx_y, indx_x]
                        delta = np.polynomial.polynomial.polyval(
                            gt_depth, stddev_poly[indx_y, indx_x])

                        # Draw vertical line denoting measurement
                        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                        ax_datas[i].vlines(
                            d_meas, 0, ylims[1], linestyles='dashed', alpha=1.0, lw=1)
                        ax_datas[i].annotate(s='$d_{meas}$',  # = '+('%.2f' % d_meas) +'$',
                                             xy=((
                                                (d_meas-xlims[0] - delta)/(xlims[1]-xlims[0])), 1.01),
                                             xycoords='axes fraction',
                                             verticalalignment='right',
                                             horizontalalignment='right bottom',
                                             rotation=0)
                        ax_datas[i].vlines(z_distances[np.argmax(w_is)], 0, ylims[1], cycle[0],
                                           linestyles='-', alpha=1.0, lw=1)
                        ax_datas[i].annotate(s='$d_{\omega_{max}}$',  # = '+('%.2f' % d_meas) +'$',
                                             xy=((
                                                (z_distances[np.argmax(w_is)]-xlims[0]-delta)/(xlims[1]-xlims[0])), 1.01),
                                             xycoords='axes fraction',
                                             verticalalignment='right',
                                             horizontalalignment='right bottom',
                                             rotation=0)
                        ax_datas[i].vlines(gt_depth, 0, ylims[1], cycle[3],
                                           linestyles='-', alpha=1.0, lw=2)
                        ax_datas[i].annotate(s='$d_{gt}$',  # = '+('%.2f' % d_meas) +'$',
                                             xy=((
                                                (gt_depth-xlims[0]-delta)/(xlims[1]-xlims[0])), 1.01),
                                             xycoords='axes fraction',
                                             verticalalignment='right',
                                             horizontalalignment='right bottom',
                                             rotation=0)
                        ax_datas[i].grid(b=True, which='major', color='k',
                                         linestyle=':', alpha=0.8)
                        ax_datas[i].grid(b=True, which='minor', axis='x',
                                         color='k', linestyle=':', alpha=0.1)
                        ax_datas[i].minorticks_on()
                        # extraticks = [d_meas]
                        # ax_datas[i].set_xticks(list(ax_datas[i].get_xticks())[
                        #     1:-1] + extraticks)
                        # ax_flipped = ax_datas[i].twiny()
                        # ax_flipped.set_xticks([d_meas])
                        # ax_flipped.set_xticklabels(str(d_meas))

                        # Also add a patch to show the 3 sigma bounds

                        ax_datas[i].add_patch(
                            Rectangle((gt_depth - delta, 0.0), 2*delta, ylims[1], alpha=0.2, ec='black', ls=':', fc=cycle[3]))
                        ax_datas[i].set_xlabel('Cell Depth $||d_i||$')

                        # Show 'explosion line'
                        con1 = ConnectionPatch(xyA=(event.xdata, event.ydata), xyB=(xlims[0], ylims[1]), coordsA="data", coordsB="data",
                                               axesA=ax_imgs[i], axesB=ax_datas[i], color="black", ls=':')
                        con2 = ConnectionPatch(xyA=(event.xdata, event.ydata), xyB=(xlims[1], ylims[1]), coordsA="data", coordsB="data",
                                               axesA=ax_imgs[i], axesB=ax_datas[i], color="black", ls=':')
                        # First remove existing connecting patch
                        for obj in ax_imgs[i].findobj(match=type(con1)):
                            obj.remove()
                        ax_imgs[i].add_artist(con1)
                        ax_imgs[i].add_artist(con2)

                        ax_datas[i].legend()

                    # Draw circle on image
                    for ax_handle in [ax_octo, ax_mrf]:
                        for obj in ax_handle.findobj(match=type(plt.Circle((1, 1)))):
                            obj.remove()
                        circle = plt.Circle(
                            (event.xdata, event.ydata), 2, color='red')
                        ax_handle.add_artist(circle)
                    fig.tight_layout()
                    fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

accs = np.array(accs)
print np.mean(accs, axis=0)
print np.std(accs, axis=0)

rot_t = 1.0
trans_t = 1.0
suffix = '_res_' + str(dtype(params.res)).replace('.', '_') + '_rot_' + \
    str(dtype(rot_t)).replace('.', '_') + '_trans_' + str(dtype(trans_t)).replace('.', '_')
if params.use_polys:
    suffix += 'poly'
np.save(main_dir+dataset+'/generated/octomap' + suffix + '_accuracies.npy', accs)