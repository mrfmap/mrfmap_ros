#!/usr/bin/env python
import numpy as np
from mrfmap_ros.GVDBPyModules import GVDBInference, GVDBImage, gvdb_params, GVDBMapLikelihoodEstimator, KeyframeSelector, GVDBOctomapWrapper, PangolinViewer
from mrfmap_ros.MRFMapRosPyModules import GVDBBatchMapCreator
# from plotting_utils.BagReader import BagReader
import tf
import pdb
from geometry import se3, SE3
import cv2
import rospy
# import glob
import rosbag
import yaml
import os
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import cv_bridge
import click
# from test_against_octomap import OctomapViewer, OctomapGenerator
from mrfmap_ros.MRFMapGenerator import MRFMapGenerator
import matplotlib.pyplot as plt
import time

dtype = np.float32
main_dir = '/home/icoderaven/bagfiles/'
dataset = 'ICL'
bagfile = main_dir+dataset+'/living_room_with_rgb_noise.bag'
config_file = '../../config/ICL.yaml'
images = []
bridge = cv_bridge.CvBridge()
with rosbag.Bag(bagfile, 'r') as bag:
  for topic, msg, stamp in bag.read_messages('/camera/depth/noisy_image'):
        depth_img = bridge.imgmsg_to_cv2(msg, "32FC1").astype(dtype)
        if msg.encoding == "16UC1":
            depth_img /= 1000.0
        images.append(depth_img)
        if len(depth_img) > 10:
            break

params = gvdb_params()
params.load_from_file(config_file)
params.set_from_python()

with open(config_file) as f:
    cam_in_body = np.array(yaml.load(f)['viewer_params_nodes']['cam_in_body'], dtype=dtype).reshape(4,4)

pose = np.eye(4, dtype=dtype)    
pose_cpp = np.dot(pose, cam_in_body).astype(dtype)

# inference = GVDBInference(True, False)
# pango_viewer = PangolinViewer("MyViewer", inference)
pango_viewer = PangolinViewer("MyViewer")
# inference.add_camera_with_depth(pose, images[0])
pango_viewer.add_keyframe(pose_cpp, images[0])

time.sleep(2)
delta_pose = np.eye(4, dtype=dtype)
delta_pose[:3,3] = [3,0,0]

updated_pose_cpp = np.dot(delta_pose, pose_cpp)
pango_viewer.set_kf_pose(0, updated_pose_cpp)
pango_viewer.set_latest_pose(updated_pose_cpp)

time.sleep(2)
pango_viewer.set_kf_pose(0, pose_cpp)
pango_viewer.set_latest_pose(pose_cpp)


time.sleep(2)
del pango_viewer