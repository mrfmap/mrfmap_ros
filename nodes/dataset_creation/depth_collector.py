#!/usr/bin/env python2.7
import numpy as np
import sys
import cv2
import tf
import pdb
import yaml
import os.path
import glob
from geometry import SE3, se3
import click
import matplotlib.pyplot as plt
import rosbag
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('seaborn-muted')

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

from mrfmap_ros.GVDBPyModules import gvdb_params
from mrfmap_ros.MRFMapRosPyModules import DepthCalibrator

from mayavi import mlab

class DataVisualiser:
    def __init__(self, params_config_file):
        self.shutdown = False
        with open(params_config_file) as f:
            node = yaml.load(f)
            gvdb_node = node['gvdb_params']
            self.K = np.array(gvdb_node['K']).reshape(3,3)
            self.rows = gvdb_node['rows']
            self.cols = gvdb_node['cols']
            self.poly_degree = gvdb_node['poly_degree'] - 1
            checkerboard_node = node['checkerboard_params']
            self.board_rows = checkerboard_node['rows']
            self.board_cols = checkerboard_node['cols']
            self.board_s = checkerboard_node['s']
            self.tag_to_board = np.array(checkerboard_node['tag_in_board']).reshape(4,4)
            viewer_node = node['viewer_params_nodes']
            self.body_to_cam = np.linalg.inv(np.array(viewer_node['cam_in_body']).reshape(4,4))
            self.img_scale_factor = viewer_node['img_scale_factor']
        params = gvdb_params()
        params.load_from_file(params_config_file)
        params.set_from_python()

        self.tag_in_cam = np.eye(4)
        #ROS stuff
        self.bridge = CvBridge()
        self.temp = None

        self.depth_max = 2.5 # meters
        self.depth_min = 0.25 # meters
        self.volume_cell_size = 0.1 # meters
        self.cell_width = 20 #pixels
        self.cell_height = 20 #pixels
        # Get the bounds of the frustum
        # get_x_lim = lambda z : ([0, self.cols] - self.K[0,2]) *z/(self.K[0,0])
        # get_y_lim = lambda z : ([0, self.rows] - self.K[1,2]) *z/(self.K[1,1])
        z_lim = [self.depth_min, self.depth_max]
        self.z_bin_stride =  self.volume_cell_size
        self.z_bin_max = 500

        self.calibrator = DepthCalibrator(self.cell_width, self.cell_height, self.depth_min, self.depth_max, self.z_bin_stride, self.z_bin_max)
        self.num_z_bins = np.floor((self.depth_max -  self.depth_min)/self.z_bin_stride)
        self.z_ranges = np.arange(z_lim[0], z_lim[1], self.z_bin_stride) + self.z_bin_stride/2.0 # To get center points of bins
        self.xs = np.array([])
        self.ys = np.array([])
        self.zs = np.array([])
        self.counts = np.array([])
        for z_depth in self.z_ranges:
            x_cells = (np.arange(0, self.cols, self.cell_width) - self.K[0,2] + 0.5 * self.cell_width) * z_depth/self.K[0,0]
            y_cells = (np.arange(0, self.rows, self.cell_height) - self.K[1,2] + 0.5 * self.cell_height) * z_depth/self.K[1,1]
            xy_x_cells, xy_y_cells = np.meshgrid(x_cells, y_cells)

            count_vals_xy = np.ones_like(xy_x_cells)*z_depth*100

            self.xs = np.append(self.xs, xy_x_cells.ravel())
            self.ys = np.append(self.ys, xy_y_cells.ravel())
            self.zs = np.append(self.zs, z_depth * np.ones_like(xy_y_cells.ravel()))

            self.counts = np.append(self.counts, count_vals_xy)

        self.fig = mlab.figure()
        self.frustum = mlab.points3d(self.xs, self.ys, self.zs, (self.z_bin_max + 1 - self.counts)/self.z_bin_max, mode='cube',
        scale_mode='scalar',
        scale_factor= 0.05, #self.volume_cell_size,
        transparent=False,
        colormap='viridis'
        )

        # mlab.colorbar()

        # Also display a plane at checkerboard coordinates
        self.tag_pts = np.array([[-1.0,-1,0,1],
         [self.board_cols,-1,0,1],
          [self.board_cols,self.board_rows,0,1],
          [-1,self.board_rows,0,1],
           [-1,-1,0,1]])
        self.tag_pts[:, :2] *= self.board_s
        checkerboard_pts = np.dot(self.tag_in_cam, self.tag_pts.T).T
        self.plane = mlab.plot3d(checkerboard_pts[:,0], checkerboard_pts[:,1], checkerboard_pts[:,2], color=(1,0,0), line_width=10.0, figure=self.fig)

        # Orient camera to custom view
        self.fig.scene.camera.position = [12.38772303523827, -5.0004336639288685, -6.50689252170525]
        self.fig.scene.camera.focal_point = [0.9927493097237332, -0.6784060264418522, 3.3856669858062336]
        self.fig.scene.camera.view_angle = 30.0
        self.fig.scene.camera.view_up = [-0.12302103915930133, -0.9535682261777465, 0.2749062784812164]
        self.fig.scene.show_axes = True

    def populate_with_existing_data(self, file_name):
        self.calibrator.read_data(file_name)

    def run_loop(self):
        @mlab.animate(delay=100)
        def anim():
            f = mlab.gcf()
            while not self.shutdown:
                # Get the zbindata
                counts = np.array([])
                for w,z in enumerate(self.z_ranges):
                    # Only show z slices in a +-1 z bin range
                    z_index = int(np.ceil((self.tag_in_cam[2,3] - self.depth_min)/(self.z_bin_stride)))
                    bin_counts = np.zeros( int(np.ceil(1.0*self.cols/self.cell_width)) * int(np.ceil(1.0*self.rows/self.cell_height) ), dtype=np.int32)
                    if np.abs(z_index - w) < 2:
                        self.calibrator.get_zbin_counts(w, bin_counts)
                    counts = np.append(counts, bin_counts)

                self.frustum.mlab_source.set(scalars= 0.1+(counts)/self.z_bin_max)
                # Get board points from pose
                checkerboard_pts = np.dot(self.tag_in_cam, self.tag_pts.T).T
                # pdb.set_trace()
                self.plane.mlab_source.set(x = checkerboard_pts[:,0], y = checkerboard_pts[:,1], z = checkerboard_pts[:,2])
                if self.temp is not None:
                    cv2.imshow('temp', self.temp)
                    cv2.waitKey(10)
                yield
        anim()
        mlab.show()

    def transform_matrix_from_odom(self, msg):
        translation = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        quaternion = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                msg.pose.orientation.z, msg.pose.orientation.w])
        T = tf.transformations.quaternion_matrix(quaternion)
        T[:3, 3] = translation
        return T

    def got_tuple(self, depth_img_msg, board_odom, body_odom, gray_img_msg):
        depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "32FC1").copy()
        gray_img = self.bridge.imgmsg_to_cv2(gray_img_msg, "passthrough").copy()

        depth_img *= self.img_scale_factor
        depth_img[np.isnan(depth_img)] = 0.0
        board_to_world = self.transform_matrix_from_odom(board_odom)
        body_to_world = self.transform_matrix_from_odom(body_odom)
        # Get the tag pose in the camera frame using the extrinsics
        self.tag_in_cam = np.dot(self.body_to_cam , np.dot(np.linalg.inv(body_to_world), np.dot(board_to_world, self.tag_to_board)))
        checkerboard_pts = np.dot(self.tag_in_cam, self.tag_pts.T).T
        projected = np.dot(self.K, checkerboard_pts[:,:3].T)
        projected /= projected[2]
        projected = projected[:2].T
        # Use these pts to create a masked depth image
        mask_depth_img = np.zeros((depth_img.shape[0], depth_img.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask_depth_img, projected.astype(np.int32), (255,255,255))
        masked_depth_img = depth_img.copy()
        masked_depth_img[mask_depth_img == 0] = 0.0
        self.temp = gray_img.copy()
        self.temp[masked_depth_img == 0] = 0

        # Push this data to the DepthCalibrator class
        # TODO: check if currently busy...
        self.calibrator.add_image(masked_depth_img.astype(np.float32), self.tag_in_cam.astype(np.float32))

    def visualise_measured_depth(self, save_path):
        col_s = int(np.ceil(1.0*self.cols/self.cell_width))
        row_s = int(np.ceil(1.0*self.rows/self.cell_height))
        bias_image = np.zeros((row_s, col_s, len(self.z_ranges)))
        noise_image = np.zeros((row_s, col_s,  len(self.z_ranges)))
        vis_img = np.zeros((row_s, col_s), dtype=np.int32)
        print 'Collecting data...'
        for w,z in enumerate(self.z_ranges):
            bin_counts = np.zeros(row_s*col_s, dtype=np.int32)
            self.calibrator.get_zbin_counts(w, bin_counts)
            vis_img += bin_counts.reshape(row_s, col_s)
            for row in range(row_s):
                for col in range(col_s):
                    data = np.zeros(2*bin_counts[row*col_s + col], dtype=np.float32)
                    self.calibrator.get_zbin_data(row, col, w, data)
                    measured_dists = data[0::2]
                    gt_dists = data[1::2]
                    # Sort both by increasing gt
                    indices = gt_dists.argsort()
                    gt_dists = gt_dists[indices]
                    measured_dists = measured_dists[indices]
                    # Store mean and variance within this bin at this (super)pixel location
                    bias_image[row, col, w] = np.nanmean(measured_dists[indices] - gt_dists[indices])
                    noise_image[row, col, w] = np.nanstd(measured_dists[indices] - gt_dists[indices])

        # Compute the bias and the variance polynomials for every superpixel
        bias_poly = np.zeros((row_s, col_s, self.poly_degree+1))
        stddev_poly = np.zeros((row_s, col_s, self.poly_degree+1))

        print 'Fitting polynomials...'
        for row in range(row_s):
            for col in range(col_s):
                idx = np.isfinite(bias_image[row, col])
                bias_poly[row, col] = np.polynomial.polynomial.polyfit(
                    self.z_ranges[idx], bias_image[row, col][idx], self.poly_degree)
                idx = np.isfinite(noise_image[row, col])
                stddev_poly[row, col] = np.polynomial.polynomial.polyfit(
                    self.z_ranges[idx], noise_image[row, col][idx], self.poly_degree)

        fig = plt.figure(0)
        ax_grid_img = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        ax_data = plt.subplot2grid((2, 2), (0, 1), colspan=1)
        ax_residuals = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        # Show a visualisation of data points collected
        from matplotlib.colors import LogNorm
        ax_grid_img_handle = ax_grid_img.imshow(
            vis_img, interpolation='none', norm=LogNorm(vmin=vis_img.min()+1, vmax=vis_img.max()))
        divider = make_axes_locatable(ax_grid_img)
        cax = divider.append_axes('right', size='10%', pad=0.1)
        # t = [0.01, 0.1, 1.0, 10.0]
        # , ticks=t, format='%.2f')
        fig.colorbar(ax_grid_img_handle, cax=cax, orientation='vertical')
        fig.tight_layout()

        ax_grid_img.set_title('Grid Data Counts')

        def on_click(event):
            x = event.xdata.astype(np.int)
            y = event.ydata.astype(np.int)

            if event.inaxes is ax_grid_img:
                size = self.calibrator.get_bin_size(y, x)
                data = np.zeros(size, dtype=np.float32)
                self.calibrator.get_bin_data(y, x, data)

                gt_dists = data[1::2]
                measured_dists = data[0::2]

                # Sort both by increasing gt
                indices = gt_dists.argsort()
                gt_dists = gt_dists[indices]
                measured_dists = measured_dists[indices]

                # Draw circle on image
                for obj in ax_grid_img.findobj(match=type(plt.Circle((1, 1)))):
                    obj.remove()
                circle = plt.Circle(
                    (event.xdata, event.ydata), 1, color='red')
                ax_grid_img.add_artist(circle)

                ax_data.clear()
                ax_data.plot(gt_dists, measured_dists, '.')
                xss = np.linspace(np.min(gt_dists), np.max(gt_dists), 10)
                ax_data.plot(xss, xss, '--')
                # Also fit a quadratic polynomial to this data
                fit = np.polynomial.polynomial.polyfit(
                    gt_dists, measured_dists, self.poly_degree)
                ax_data.plot(
                    xss, np.polynomial.polynomial.polyval(xss, fit), 'r')

                ax_data.grid(b=True, which='major', color='k',
                             linestyle=':', alpha=0.8)
                ax_data.minorticks_on()
                ax_data.set_xlabel('Ground Truth Depth (m)')
                ax_data.set_ylabel('Measured Depth (m)')

                # Find the residuals to the polynomial fit
                # residuals = (
                #     measured_dists - np.polynomial.polynomial.polyval(gt_dists, fit))**2
                ax_residuals.clear()
                # ax_residuals.plot(gt_dists, residuals)
                residuals = measured_dists - gt_dists
                ax_residuals.plot(gt_dists, residuals, '.')

                xs = np.linspace(self.depth_min, self.depth_max, 100)
                ax_residuals.plot(xs, np.polynomial.polynomial.polyval(
                    xs, bias_poly[y, x]), 'r')
                ys = np.polynomial.polynomial.polyval(xs, bias_poly[y, x])
                # ax_residuals.plot(distances, np.polynomial.polynomial.polyval(distances, stddev_poly[y,x]), 'k')
                stddevs = np.polynomial.polynomial.polyval(
                    xs, stddev_poly[y, x])
                ax_residuals.fill_between(
                    xs, ys-3*stddevs, ys+3*stddevs, alpha=0.2)
                ax_residuals.set_xlabel('Distance')
                ax_residuals.set_ylabel('Bias/Variance')

                fig.tight_layout()
                fig.canvas.draw()
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        fig.tight_layout()
        plt.show()
        # Save polynomials?
        if raw_input('Save polynomials? y/n') in ['y', 'Y']:
            np.save(save_path+'_bias.npy', bias_image)
            np.save(save_path+'_stddev.npy', noise_image)
            with open(save_path+'_bias.yml', 'w') as f:
                yaml.dump(bias_poly.ravel().tolist(), f)
            with open(save_path+'_stddev.yml', 'w') as f:
                yaml.dump(stddev_poly.ravel().tolist(), f)


if __name__ == "__main__":
    params_config_file = '../../config/realsense_848.yaml'
    visualiser = DataVisualiser(params_config_file)
    save_path = 'realsense_848_small_board_' + \
        str(np.float32(visualiser.depth_min)).replace('.', '_')+'_to_'+ \
        str(np.float32(visualiser.depth_max)).replace('.', '_')+'_at_'+ \
        str(np.float32(visualiser.z_bin_stride)).replace('.', '_')+'_res_'+ \
        str(visualiser.cell_width)+'_x_'+ \
        str(visualiser.cell_height)
    data_path = save_path+'.dat'

    board_pose_topic = '/vicon/small_checkerboard'
    body_pose_topic = '/vicon/realsense_rig_new'
    depth_img_topic = '/camera/aligned_depth_to_infra1/image_raw'
    gray_img_topic = '/camera/infra1/image_rect_raw'

    topics_to_parse = [depth_img_topic, board_pose_topic, body_pose_topic, gray_img_topic]
    subs = []
    subs.append(Subscriber(topics_to_parse[0], Image))
    subs.append(Subscriber(topics_to_parse[1], PoseStamped))
    subs.append(Subscriber(topics_to_parse[2], PoseStamped))
    subs.append(Subscriber(topics_to_parse[3], Image))

    synchronizer = ApproximateTimeSynchronizer(subs, 10, 0.05)

    synchronizer.registerCallback(visualiser.got_tuple)

    rospy.init_node('depth_collector')
    rospy.Subscriber(topics_to_parse[0], Image,
                    lambda msg: subs[0].signalMessage(msg))
    rospy.Subscriber(topics_to_parse[1], PoseStamped,
                    lambda msg: subs[1].signalMessage(msg))
    rospy.Subscriber(topics_to_parse[2], PoseStamped,
                    lambda msg: subs[2].signalMessage(msg))
    rospy.Subscriber(topics_to_parse[3], Image,
                    lambda msg: subs[3].signalMessage(msg))
    # Populate with existing data
    if os.path.exists(data_path):
        if raw_input('Found existing file. Load? y/n') in ['y', 'Y']:
            visualiser.populate_with_existing_data(data_path)
    # Start display loop
    visualiser.run_loop()
    # Ask to save the data?
    if raw_input('Save? y/n') in ['y', 'Y']:
        print 'saving to '+data_path
        visualiser.calibrator.save_data(data_path)
    print 'Done! Ready to quit now (Ctrl+C)'
    visualiser.visualise_measured_depth(save_path)