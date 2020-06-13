#!/usr/bin/env python
import argparse
import yaml
import cv2
import zlib
import numpy as np

import rosbag
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
import tf.transformations
import pdb

# For progressbar
import click


def read_stanford_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            # Assuming that the metadata is already lined up + synchronised...
            metadata = map(int, metastr.split())
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(mat)
            metastr = f.readline()
    return traj


def read_tum_trajectory(filename):
    # TODO
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Convert a png dataset to a compressed ROS bag.")
    parser.add_argument('--input', help='Input directory path', required=True)
    parser.add_argument('--rgb', help='Directory with RGB images', required=True)
    parser.add_argument('--depth', help='Directory with depth images', required=True)
    parser.add_argument('--trajectory', help='Trajectory file name', required=True)
    parser.add_argument('--type', help='Dataset type TUM/Stanford', required=True)
    parser.add_argument('--cam_yml', help='YAML file with K in ROS format', required=True)
    parser.add_argument('--visualise', help='Visualise image', default=False)
    parser.add_argument('--namespace', help='camera namespace', default='/camera')

    args = parser.parse_args()

    print "Converting images and trajectory in %s to compressed ROS bag file..." % (
        args.input)
    print 'Dataset type is ' + args.type

    bag = rosbag.Bag(args.input + '/' + args.input.split('/')[-1] + '.bag', mode='w', compression='lz4')
    bridge = CvBridge()

    # Read in YAML file
    with open(args.input +'/' + args.cam_yml) as f:
        config = yaml.load(f)

    # Read in trajectory
    gt_traj = []
    if args.type == 'Stanford':
        gt_traj = read_stanford_trajectory(args.input + '/' + args.trajectory)
        prev_stamp = 0.0

    try:
        with click.progressbar(range(len(gt_traj))) as bar:
            for count in bar:
                if args.type == 'Stanford':
                    # The time stamp is just set at 30 Hz increments...
                    prev_stamp = prev_stamp + 1.0/30
                    event_stamp = rospy.Time().from_sec(prev_stamp)

                    # Read RGB image
                    rgb_img = cv2.imread(args.input + '/' + args.rgb + ('/%06d' % (count+1)) + '.png')
                    if args.visualise:
                        cv2.imshow('RGB', rgb_img)
                        cv2.waitKey(10)

                    depth_img = cv2.imread(args.input + '/' + args.depth + ('/%06d' % (count+1)) + '.png', cv2.IMREAD_UNCHANGED)

                rgb_ros_msg = bridge.cv2_to_imgmsg(rgb_img, 'bgr8')
                rgb_ros_msg.header.seq = (count+1)

                rgb_ros_msg.header.stamp = event_stamp
                rgb_ros_msg.header.frame_id = args.namespace

                depth_ros_msg = bridge.cv2_to_imgmsg(depth_img)
                depth_ros_msg.header.seq = (count+1)
                depth_ros_msg.header = rgb_ros_msg.header

                camera_info = CameraInfo()
                camera_info.header = rgb_ros_msg.header
                camera_info.height = rgb_img.shape[0]
                camera_info.width = rgb_img.shape[1]

                camera_info.distortion_model = 'plumb_bob'
                camera_info.K = config["K"]

                pose_ros_msg = PoseStamped()
                pose_ros_msg.header.stamp = event_stamp
                pose_ros_msg.header.seq = count

                pose_ros_msg.pose.position.x = gt_traj[count][0,3]
                pose_ros_msg.pose.position.y = gt_traj[count][1,3]
                pose_ros_msg.pose.position.z = gt_traj[count][2,3]

                quat = tf.transformations.quaternion_from_matrix(gt_traj[count])
                pose_ros_msg.pose.orientation.w = quat[3]
                pose_ros_msg.pose.orientation.x = quat[0]
                pose_ros_msg.pose.orientation.y = quat[1]
                pose_ros_msg.pose.orientation.z = quat[2]

                bag.write(args.namespace+'/rgb/image_raw', rgb_ros_msg, event_stamp)
                bag.write(args.namespace+'/depth/image_rect', depth_ros_msg, event_stamp)
                bag.write(args.namespace+'/pose', pose_ros_msg, event_stamp)
                if count < 10:
                    bag.write(args.namespace+'/camera_info', camera_info, event_stamp)
    finally:
        bag.close()
        print("Done.")

if __name__ == '__main__':
    main()