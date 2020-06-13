#!/usr/bin/env python
import numpy as np
from mrfmap_ros.MRFMapGenerator import MRFMapGenerator
from mrfmap_ros.GVDBPyModules import GVDBInference, GVDBImage, gvdb_params, PangolinViewer, KeyframeSelector
import pdb
from geometry import se3, SE3
import cv2
import glob
import yaml

dtype=np.float32

from time import time
_tstart_stack = []


def tic():
    _tstart_stack.append(time())


def toc(fmt="Elapsed: %s s"):
    print fmt % (time() - _tstart_stack.pop())


show_map = False
show_pango_viewer = True
vis_likelihoods = True
plot_1d = False
plot_2d = False

# Note: This script is only useful when MRFMap is compiled with DEBUG enabled!

if show_map:
    from mayavi import mlab


if __name__ == "__main__":
    main_dir = '/home/icoderaven/bagfiles/'
    datasets = ['coffin_world_640']
    dataset = datasets[0]
    config_dir = '../../config/'
    params_config_file = config_dir+dataset+'.yaml'
    use_multiplicative_noise = False

    noise_val = 0.1
    noise_str = str(noise_val).replace('.', '_')
    # hfov = 0.9799151
    img_dims = np.array([640, 480])
    # img_dims = np.array([320, 240])
    use_realsense_noise = False
    if use_realsense_noise:
        # Load the polynomials
        with open(config_dir+'realsense_bias_poly5x.yml') as f:
            bias_poly = np.array(yaml.load(f)).reshape(
                img_dims[1]/20, img_dims[0]/20, 3)
            # Now just repeat to img dims for easy noisy image generation
            bias_poly = np.repeat(np.repeat(bias_poly, 20, axis=0), 20, axis=1)
        with open(config_dir+'realsense_stddev_poly5x.yml') as f:
            stddev_poly = np.array(yaml.load(f)).reshape(
                img_dims[1]/20, img_dims[0]/20, 3)
            # Now just repeat to img dims for easy noisy image generation
            stddev_poly = np.repeat(
                np.repeat(stddev_poly, 20, axis=0), 20, axis=1)

    downsample_image = False
    downsample_times = 1

    # path_str = 'data/res_640x480/'
    # path_str = 'data/res_640_more/'
    # path_str = 'data/res_640x480_fixed/'
    # path_str = 'data/airsim_factory/'
    path_str = main_dir + dataset + '/'

    # Determine the number of cameras inside this directory
    if use_multiplicative_noise:
        file_list = glob.glob(path_str + 'depth_noise_' +
                              noise_str + '/0_depth_*.npy')
    else:
        file_list = glob.glob(path_str + 'depth_*.npy')
    num_images = len(file_list)
    # num_images = 1
    print 'Detected {0} images'.format(num_images)
    cam_id = 0

    map_indices = []

    tic()

    print 'Creating params...'
    if downsample_image:
        img_dims = img_dims/(2**downsample_times)

    if glob.glob(params_config_file) == []:
        print 'No params_config_file!, bailing'
        exit

    mrfmap = MRFMapGenerator(
        params_config_file, main_dir+dataset+'/generated/', 'mrfmap')
    mrfmap.inference = GVDBInference(True, False)

    if show_pango_viewer:
        pango_viewer = PangolinViewer("MyViewer", mrfmap.inference)
    print 'Done!'

    poses = []
    depth_imgs = []
    gt_depth_imgs = []
    image_ptrs = []
    # list_of_images = [0,1,2,11,10]
    list_of_images = range(num_images)
    num_images = 0
    mrfmap.inference.set_selected_x_y(338, 204)

    for image_index in list_of_images:
        pose = np.load(path_str+'pose_' + str(image_index) + '.npy')
        cam_in_body = np.eye(4)
        cam_in_body[:3, :3] = np.array(
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        pose_cpp = np.dot(pose, cam_in_body).astype(dtype)

        origin = pose_cpp[:3, 3]
        camera_normal = pose_cpp[:3, 2]

        if use_multiplicative_noise:
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
            # Now get stddev values
            stddev_depth_img = stddev_poly[:, :, 2] * depth_img_cpp**2 + \
                stddev_poly[:, :, 1]*depth_img_cpp + stddev_poly[:, :, 0]
            # Now sample from stddev values and add to biased depth image
            generated_img = biased_depth_img + stddev_depth_img * \
                np.random.randn(img_dims[1], img_dims[0])
            # pdb.set_trace()
            # Assign this to the depth_img_cpp
            depth_img_cpp = generated_img.astype(dtype)

        if downsample_image:
            for i in range(downsample_times):
                # Downsample the image
                # depth_img_cpp = cv2.pyrDown(depth_img_cpp)
                depth_img_cpp = cv2.resize(
                    depth_img_cpp, (depth_img_cpp.shape[1]/2, depth_img_cpp.shape[0]/2), interpolation=cv2.INTER_NEAREST)

        if mrfmap.keyframe_selector.is_keyframe(pose_cpp.astype(dtype)):
            poses.append(pose_cpp)
            depth_imgs.append(depth_img_cpp)
            image_ptrs.append(GVDBImage(depth_img_cpp))
            mrfmap.add_data(pose_cpp.astype(dtype), image_ptrs[-1])
            if show_pango_viewer:
                pango_viewer.add_keyframe(
                    pose_cpp.astype(dtype), depth_img_cpp)
            num_images += 1
        else:
            print 'oof, not a keyframe'
            if show_pango_viewer:
                pango_viewer.add_frame(pose_cpp.astype(dtype), depth_img_cpp)

    toc()
    
    # mrfmap.save_map()

    if show_map:
        mrfmap.show_map()

    if vis_likelihoods:
        # VISUALIZE LIKELIHOODS
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import CheckButtons
        plt.style.use('seaborn-muted')

        image = np.zeros(depth_imgs[0].shape, dtype=dtype)
        mrfmap.inference.get_likelihood_image(cam_id, image)
        expected_depth_image = np.zeros(depth_imgs[0].shape, dtype=dtype)
        mrfmap.inference.get_expected_depth_image(cam_id, expected_depth_image)
        # fig2 = plt.figure(2)
        # ax_1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        # ax_2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)
        # ax_1.imshow(depth_imgs[cam_id])
        # ax_2.imshow(expected_depth_image)
        # plt.show()
        img = np.exp(image)

        fig = plt.figure(0)
        ax_like_img = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        ax_depth_img = plt.subplot2grid((2, 2), (0, 1), colspan=1)
        ax_like_imshow_handle = ax_like_img.imshow(img, picker=True)
        ax_depth_imshow_handle = ax_depth_img.imshow(
            depth_imgs[cam_id], picker=True)
        ax_data = plt.subplot2grid((2, 2), (1, 0), colspan=2, picker=True)

        rax = plt.axes([0.85, 0.6, 0.2, 0.4])
        labels_right = [
            ['ray_mu_d_i', False],
            ['last_outgoing', False],
            ['pos_log_msg_sum', False],
            ['neg_norm', True],
            ['pos_msg', True],
            ['length', False],
            ['pos_iso_msg', False],
            ['prod_mu_o_0s', False]
        ]
        right_check = CheckButtons(rax, tuple(
            [label[0] for label in labels_right]),
            tuple([label[1] for label in labels_right]))

        ray_idx = [None]
        voxel_ids = [None]
        node_ids = [None]
        selected_node = None
        brick_ids = [None]
        z_distances = [None]
        click_mode = None

        def onclick(event):
            global voxel_ids
            global z_distances
            global node_ids
            global brick_ids
            global selected_node
            global atlas_ids
            global click_mode

            indx_x = (event.xdata + 0.5).astype(np.int)
            indx_y = (event.ydata + 0.5).astype(np.int)
            print indx_x, indx_y
            # print b_is, z_given_b_is
            if event.inaxes is ax_data:
                # We're in the data plot, clicking should open up the window that
                # pushes out messages being sent by the rays
                data = ax_data.get_lines()[0].get_data()
                ind = np.searchsorted(data[0], event.xdata)
                if click_mode == 1:
                    vox_id = voxel_ids[ind]
                elif click_mode == 3:
                    vox_id = atlas_ids[ind]
                print 'Index is '+str(ind) + ' voxel id is ' + \
                    str(vox_id) + ' dist is '+str(data[0][ind])
                mrfmap.inference.set_selected_voxel(vox_id)
                selected_node = [node_ids[ind], brick_ids[:, ind], vox_id]
                # Perform  mrfmap.inference again to print output?
                mrfmap.inference.perform_inference_dryrun()
                # Also show the line for selected voxel
                ax_data.vlines(z_distances[ind], 0,
                               1.0, linestyles='dotted', alpha=0.8)

            elif event.inaxes is ax_like_img:

                # Set the x and y
                mrfmap.inference.set_selected_x_y(indx_x, indx_y)
                # We should also reset selected node
                mrfmap.inference.set_selected_voxel(0)
                # Call the diagnostic method
                diag_image = np.zeros(
                    depth_imgs[0].shape, dtype=dtype).flatten()

                if event.button == 1:
                    click_mode = 1
                    mrfmap.inference.get_diagnostic_image(
                        cam_id, diag_image, 0)
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
                    ax_data.clear()
                    ax_data.plot(z_distances, alphas,
                                 label='alphas', marker='.')
                    ax_data.plot(z_distances, w_is, label='w_is', marker='.')
                    ax_data.plot(z_distances, vis_is,
                                 label='vis_is', marker='.')

                    print 'Sum of w_is is '+str(np.sum(w_is))

                elif event.button == 3:
                    click_mode = 3
                    mrfmap.inference.get_diagnostic_image(
                        cam_id, diag_image, 1)
                    # Extract data from image
                    # Get num_voxels from last entry
                    stride = 12
                    num_voxels = diag_image[-1].astype(np.int)
                    z_distances = diag_image[:stride*num_voxels:stride]
                    ray_mu_d_i = diag_image[1:stride*num_voxels:stride]
                    last_outgoing = diag_image[2:stride*num_voxels:stride]
                    pos_log_msg_sum = diag_image[3:stride*num_voxels:stride]
                    neg_norm = diag_image[4:stride*num_voxels:stride]
                    pos_msg = diag_image[5:stride*num_voxels:stride]
                    length = diag_image[6:stride*num_voxels:stride]
                    pos_iso_msg = diag_image[7:stride*num_voxels:stride]
                    atlas_ids = np.frombuffer(diag_image, dtype=np.uint64)[
                        4:num_voxels*(stride/2):(stride/2)]
                    prod_mu_o_0s = diag_image[10:stride*num_voxels:stride]

                    dict_labels = {'ray_mu_d_i': ray_mu_d_i,
                                   'last_outgoing': last_outgoing,
                                   'pos_log_msg_sum': pos_log_msg_sum,
                                   'neg_norm': neg_norm,
                                   'pos_msg': pos_msg,
                                   'length': length,
                                   'pos_iso_msg': pos_iso_msg,
                                   'prod_mu_o_0s': prod_mu_o_0s}
                    # Plot corresponding data
                    ax_data.clear()
                    for label in labels_right:
                        if label[1]:
                            ax_data.plot(z_distances, dict_labels[label[0]],
                                         label=label[0], marker='.')

                # Draw circle on image
                for obj in ax_like_img.findobj(match=type(plt.Circle((1, 1)))):
                    obj.remove()
                circle = plt.Circle(
                    (indx_x, indx_y), 1, color='red', fill=False)
                ax_like_img.add_artist(circle)

                d_meas = depth_imgs[cam_id][indx_y, indx_x]
                print 'Expected depth here is ' + \
                    str(expected_depth_image[indx_y, indx_x]
                        ) + ' actual is '+str(d_meas)

                if np.isnan(d_meas):
                    d_meas = 10.0
                xlims = ax_data.get_xlim()
                ylims = ax_data.get_ylim()
                ax_data.vlines(d_meas, 0, 1.0, 'blue',
                               linestyles='dashed', alpha=0.8)
                ax_data.annotate(s='$d_{meas}$',
                                 xy=((
                                     (d_meas-xlims[0])/(xlims[1]-xlims[0])), 1.01),
                                 xycoords='axes fraction',
                                 verticalalignment='right',
                                 horizontalalignment='right bottom',
                                 rotation=0)
                if use_realsense_noise:
                    gt_depth = gt_depth_imgs[cam_id][indx_y, indx_x]
                    gt_plus_bias = d_meas + \
                        np.polynomial.polynomial.polyval(
                            gt_depth, bias_poly[indx_y, indx_x])
                    ax_data.vlines(gt_plus_bias, 0, 1.0,
                                   linestyles='-.', alpha=0.8)
                    ax_data.annotate(s='$d_{bias}$',  # = '+('%.2f' % d_meas) +'$',
                                     xy=((
                                         (gt_plus_bias-xlims[0])/(xlims[1]-xlims[0])), 1.01),
                                     xycoords='axes fraction',
                                     verticalalignment='right',
                                     horizontalalignment='right bottom',
                                     rotation=0)
                    ax_data.vlines(gt_depth, 0, 1.0, 'red',
                                   linestyles=':', alpha=0.8)
                    ax_data.annotate(s='$d_{gt}$',  # = '+('%.2f' % d_meas) +'$',
                                     xy=((
                                         (gt_depth-xlims[0])/(xlims[1]-xlims[0])), 1.01),
                                     xycoords='axes fraction',
                                     verticalalignment='right',
                                     horizontalalignment='right bottom',
                                     rotation=0)
                    # Get the corresponding sigma
                    delta = np.polynomial.polynomial.polyval(
                        gt_depth, stddev_poly[indx_y, indx_x])
                    ax_data.add_patch(
                        Rectangle((gt_depth - delta, 0.0), 2*delta, ylims[1], alpha=0.2, ec='black', ls=':'))
                ax_data.grid(b=True, which='major', color='k',
                             linestyle=':', alpha=0.8)
                ax_data.grid(b=True, which='minor', axis='x',
                             color='k', linestyle=':', alpha=0.2)
                ax_data.minorticks_on()
                extraticks = [d_meas]
                ax_data.set_xticks(list(ax_data.get_xticks())[
                    1:-1] + extraticks)
                ax_data.set_xlabel('Cell Depth $||d_i||$')
                ax_data.legend()

            # Should also get the messages being sent along this ray
            fig.canvas.draw()

        def onpick(event):
            line = event.artist
            if not type(line) == type(plt.imshow(np.zeros((1, 1)))):
                xdata, ydata = line.get_data()
                ind = event.ind
                print 'Index is '+str(ind)
                # Get nearest voxel id
                if ind is not None:
                    # Get corresponding voxel id
                    ray_indices = VecUInt([])
                    mrfmap.inference.get_indices_along_ray(
                        0, ray_idx[0], ray_indices)
                    vox_idx = np.array(ray_indices)[ind[0]]
                    viewer.plot_rays(vox_idx, mrfmap.inference)
                    fig.canvas.draw()

        def right_check_callback(label):
            if label in [l[0] for l in labels_right]:
                idx = [l[0] for l in labels_right].index(label)
                labels_right[idx][1] = not(labels_right[idx][1])

        voxel_cube = [None for i in range(24)]

        def on_key(event):
            global cam_id
            global voxel_cube
            id_changed = False
            if event.key == 'n':
                cam_id = cam_id + 1 if cam_id < num_images - 1 else 0
                id_changed = True
            elif event.key == 'p':
                cam_id = cam_id - 1 if cam_id > 0 else num_images - 1
                id_changed = True
            elif event.key == 'i':
                # Perform 1 pass of inference
                mrfmap.inference.perform_inference()
                id_changed = True

            if id_changed:
                print 'Set cam_id to '+str(cam_id)
                # Update likelihood image displayed
                image = np.zeros(depth_imgs[0].shape, dtype=dtype)
                mrfmap.inference.get_likelihood_image(cam_id, image)
                img = np.exp(image)
                ax_like_imshow_handle.set_data(img)
                ax_depth_imshow_handle.set_data(depth_imgs[cam_id])

                if selected_node is not None:
                    # Display this selected voxel?
                    vox_coords = mrfmap.inference.get_voxel_coords(
                        [selected_node[0]]).flatten()
                    brick_id = selected_node[1]

                    # TODO: move these coords by brick index
                    render_choices = ['brick', 'voxel']

                    edges = [
                        # Back
                        (0, 1), (1, 2), (2, 3), (3, 0),
                        # Front
                        (5, 4), (4, 7), (7, 6), (6, 5),
                        # Front-to-back
                        (0, 4), (1, 5), (2, 6), (3, 7)]

                    i = 0
                    for render_choice in render_choices:
                        if render_choice is 'brick':
                            vmin = vox_coords[:3]
                            side = vox_coords[3] - vox_coords[0]
                        elif render_choice is 'voxel':
                            side = 1
                            vmin = vox_coords[:3] + brick_id

                        vertices = np.array([
                            vmin + [0, side, 0],
                            vmin + [side, side, 0],
                            vmin + [side, 0, 0],
                            vmin + [0, 0, 0],
                            vmin + [0, side, side],
                            vmin + [side, side, side],
                            vmin + [side, 0, side],
                            vmin + [0, 0, side]])

                        centered_vertices = vertices - \
                            np.array([[width], [breadth], [height]]
                                     ).transpose()/(2.0*resolution)
                        scaled_vertices = centered_vertices * resolution
                        homogenized_vertices = np.hstack(
                            (scaled_vertices, np.ones((8, 1))))
                        transformed_vertices = np.dot(np.linalg.inv(
                            poses[cam_id]), homogenized_vertices.transpose())

                        pts = np.dot(K, transformed_vertices[:3, :])
                        pts = pts[:]/pts[2, :]
                        pts = pts.transpose()

                        for e in edges:
                            if voxel_cube[i] is not None:
                                ax_depth_img.lines.remove(voxel_cube[i])
                            if render_choice is 'brick':
                                style = 'r-'
                                line_width = 0.4
                            else:
                                style = 'r-'
                                line_width = 0.9
                            voxel_cube[i], = ax_depth_img.plot(
                                (pts[e[0]][0], pts[e[1]][0]), (pts[e[0]][1], pts[e[1]][1]), style, linewidth=line_width)

                            i += 1
                        # Change the event clicked location to the mean of the projected vertices
                        # mean = pts.mean(axis=0)[:2]
                        # Find the projected center of the voxel

                    mean = (pts[3] + pts[5])/2.0
                    print 'mean of points is '+str(mean)
                    event.xdata = mean[0]
                    event.ydata = mean[1]
                    event.button = 3
                    event.inaxes = ax_like_img
                    if 0 <= event.xdata < depth_img.shape[1] and 0 <= event.ydata < depth_img.shape[0]:
                        onclick(event)
                fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        rbid = right_check.on_clicked(right_check_callback)
        kid = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    if plot_1d:
        # PLOT 1D LIKELIHOOD PLOTS
        # Now test likelihoods of perturbed camera poses
        import matplotlib.pyplot as plt

        num_poses = 11
        deltas = np.linspace(-0.1, 0.1, num_poses)
        labels = ['rx', 'ry', 'rz', 'x', 'y', 'z']
        np.set_printoptions(precision=3, suppress=True)
        axes_to_plot = range(6)
        for i in [cam_id]:  # range(num_images):
            pose_cpp = poses[i]
            depth_img_cpp = depth_imgs[i]

            sampled_poses = []
            ray_likelihoods = [[]]*6
            for j in axes_to_plot:
                print '!!Evaluating for axis '+labels[j]
                likes = []
                for k in range(num_poses):
                    print "Axis::{0} Pose::{1}".format(j, k)
                    delta_vector = np.zeros(6)
                    delta_vector[j] = deltas[k]
                    T = SE3.group_from_algebra(
                        se3.algebra_from_vector(delta_vector))
                    pose = np.dot(pose_cpp, T)
                    sampled_poses.append(pose)
                    # print 'pose is '+str(pose)
                    mrfmap.inference.set_pose(i, pose.astype(dtype))
                    # pdb.set_trace()
                    sum = mrfmap.inference.compute_likelihood(i)
                    likes.append(sum)
                ray_likelihoods[j] = np.array(likes)

            # viewer.visualize_poses(sampled_poses, resolution, scaled_dims)
            # viewer.show()

            f, ax = plt.subplots(6, 1)

            handles = []
            for j in axes_to_plot:
                handle = ax[j].plot(deltas, ray_likelihoods[j])
                handles.append(handle[0])
        f.legend(handles, labels, 'upper right')
        plt.show()

        # Compute likelihood of 2D locations with known yaw for a particular camera
        # create a square grid around ground truth location
        # import matplotlib.pyplot as plt
        # num_images = 2
        # f, ax = plt.subplots(1, num_images)

        # grid_dim = 5
        # eval_dim = 3
        # cell_res = 0.05
        # grid = np.mgrid[0:grid_dim:1, 0:grid_dim:1].reshape(2, -1).T
        # # grid = zip(range(grid_dim), range(grid_dim))
        # for cam_id in range(num_images):
        #     grid_likes = np.zeros((grid_dim*eval_dim, grid_dim*eval_dim))
        #     for item in grid:
        #         start = cell_res * \
        #             np.array([item[0] - grid_dim/2, item[1] - grid_dim/2])
        #         end = start + cell_res
        #         xys = np.mgrid[start[0]:end[0]: (end[0] - start[0])/eval_dim,
        #                        start[1]:end[1]: (end[1] - start[1])/eval_dim].reshape(2, -1).T
        #         vectors = np.hstack((xys, np.zeros((xys.shape[0], 4))))

        #         pose_cpp = poses[cam_id]
        #         depth_img_cpp = depth_imgs[cam_id]
        #         for delta_vector in vectors:
        #             T = SE3.group_from_algebra(se3.algebra_from_vector(delta_vector))
        #             pose = np.dot(pose_cpp, T)
        #             # pdb.set_trace()
        #             mrfmap.inference.add_likelihood_camera(pose.astype(dtype), hfov,
        #                                             depth_img_cpp.astype(dtype))
        #         likes = VecFloat([])
        #         mrfmap.inference.compute_ray_likelihoods(likes)
        #         grid_likes[item[0]*eval_dim:(item[0]+1)*eval_dim,
        #                    item[1]*eval_dim:(item[1]+1)*eval_dim] = np.array(likes).reshape(eval_dim, eval_dim)

        #     ax[cam_id].imshow(grid_likes, interpolation='none',
        #                       extent=[-(grid_dim/2 - 0.5 / eval_dim)*cell_res, (grid_dim/2+0.5 / eval_dim)*cell_res,
        #                               (grid_dim/2+0.5 / eval_dim)*cell_res, -(grid_dim/2 - 0.5 / eval_dim)*cell_res])
        # plt.show()
    if plot_2d:
        # Compute likelihood of 2D locations with known yaw for a particular camera
        # create a square grid around ground truth location
        # import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 2)
        axes_to_compare = [3, 5]  # first axis on x axis, second on y

        grid_dim = 4
        eval_dim = 5
        grid_res = 0.1
        cell_res = grid_res/eval_dim
        grid = np.mgrid[0:grid_dim:1, 0:grid_dim:1].reshape(2, -1).T
        grid_likes = np.zeros((grid_dim*eval_dim, grid_dim*eval_dim))
        eval_poses = [[None for _ in range(grid_dim*eval_dim)]
                      for _ in range(grid_dim*eval_dim)]
        for item in grid:
            print item
            start = grid_res * \
                np.array([item[0] - grid_dim/2, item[1] - grid_dim/2])
            end = start + grid_res
            xys = np.mgrid[start[0]:end[0]: (end[0] - start[0])/eval_dim,
                           start[1]:end[1]: (end[1] - start[1])/eval_dim].reshape(2, -1).T
            vectors = np.zeros((xys.shape[0], 6))
            vectors[:, axes_to_compare[0]] = xys[:, 1]
            vectors[:, axes_to_compare[1]] = xys[:, 0]

            pose_cpp = poses[cam_id]
            depth_img_cpp = depth_imgs[cam_id]
            likes = []
            for i, delta_vector in enumerate(vectors):
                T = SE3.group_from_algebra(
                    se3.algebra_from_vector(delta_vector))
                pose = np.dot(pose_cpp, T)
                # pdb.set_trace()
                eval_poses[item[0]*eval_dim +
                           int(i/eval_dim)][item[1]*eval_dim + i % eval_dim] = pose
                mrfmap.inference.set_pose(cam_id, pose.astype(dtype))
                # pdb.set_trace()
                sum = mrfmap.inference.compute_likelihood(cam_id)
                likes.append(sum)

            grid_likes[item[0]*eval_dim:(item[0]+1)*eval_dim,
                       item[1]*eval_dim:(item[1]+1)*eval_dim] = np.array(likes).reshape(eval_dim, eval_dim)

        ax[0].imshow(grid_likes, interpolation='none', picker=True)
        nx = grid_likes.shape[1]
        ny = grid_likes.shape[0]
        no_labels = grid_dim+1  # how many labels to see on axis x
        step_x = int(nx / (no_labels - 1))  # step between consecutive labels
        step_y = int(ny / (no_labels - 1))  # step between consecutive labels
        # pixel count at label position
        x_positions = np.arange(0, nx+1, step_x)
        # pixel count at label position
        y_positions = np.arange(0, ny+1, step_y)
        x_labels = np.linspace(-nx*cell_res/2.0, nx*cell_res/2.0,
                               nx+1)[::step_x]  # labels you want to see
        y_labels = np.linspace(-ny*cell_res/2.0, ny*cell_res/2.0,
                               ny+1)[::step_y]  # labels you want to see
        ax[0].set_xticks(x_positions)
        ax[0].set_xticklabels(x_labels.tolist())
        ax[0].set_yticks(y_positions)
        ax[0].set_yticklabels(y_labels.tolist())
        ax[0].set_xlabel('Displacement in '+labels[axes_to_compare[0]])
        ax[0].set_ylabel('Displacement in '+labels[axes_to_compare[1]])

        rect = None
        imshow_img = None

        def onclick_eval(event):
            global rect
            global imshow_img
            global cam_id
            print event.xdata, event.ydata
            x = (event.xdata + 0.5).astype(np.int)
            y = (event.ydata + 0.5).astype(np.int)
            print x, y
            if event.inaxes is ax[1]:
                # We're in the image plot, don't do anything
                pass
            else:
                # Show the likelihood image corresponding to this pixel coord
                mrfmap.inference.set_pose(cam_id, eval_poses[y][x])
                print 'Pose is '
                print eval_poses[y][x]
                img = np.zeros(depth_imgs[0].shape, dtype=dtype)
                mrfmap.inference.get_likelihood_image(cam_id, img)
                img = np.exp(img)
                if imshow_img is None:
                    imshow_img = ax[1].imshow(
                        img, picker=True, cmap='brg')
                    plt.colorbar(imshow_img)
                else:
                    imshow_img.set_data(img)
                # Maybe also highlight the specific pixel?
                if rect is not None:
                    rect.remove()

                rect = plt.Rectangle((x-0.5, y-0.5), 1, 1,
                                     edgecolor='red', facecolor='none')
                ax[0].add_artist(rect)

            # Should also get the messages being sent along this ray
            f.canvas.draw()

        cid = f.canvas.mpl_connect('button_press_event', onclick_eval)
        plt.show()

        # from mayavi import mlab
        # plot_surf = mlab.surf(grid_likes, warp_scale='auto')
        # mlab.outline(plot_surf)
        # mlab.axes(plot_surf, xlabel='x', ylabel='y')
        # mlab.show()
