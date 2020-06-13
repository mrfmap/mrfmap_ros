#!/usr/bin/env python
import argparse
import yaml
import cv2
import zlib
import numpy as np

import lcm
from xtion import rgbd_t
from microstrain import raw_t

import rosbag
import rospy
from sensor_msgs.msg import Image, CameraInfo, Imu, MagneticField
from cv_bridge import CvBridge, CvBridgeError

import pdb
import matplotlib.pyplot as plt

import click


def main():
    parser = argparse.ArgumentParser(
        description="Convert an LCM log to a ROS bag (mono/stereo images only).")
    parser.add_argument('--input', help='Input LCM log.')
    parser.add_argument('--namespace', help='camera namespace')
    parser.add_argument('--cam_yml',
                        help='Image calibration YAML file from ROS calibrator')
    parser.add_argument('--visualise', help='Visualise image', default=False)

    args = parser.parse_args()

    print "Converting images in %s to ROS bag file..." % (args.input)

    log = lcm.EventLog(args.input, 'r')
    bag = rosbag.Bag(args.input + '.converted.bag', mode='w', compression='lz4')
    bridge = CvBridge()
    
    # # Read in YAML files.
    with open(args.cam_yml) as f:
        config = yaml.load(f)
    lcm_channels = ['MICROSTRAIN_RAW', 'XTION']
    try:
        count = 0
        with click.progressbar(length=log.size()) as bar:
            for event in log:
                event_stamp = rospy.Time().from_sec(float(event.timestamp)/1e6)

                if event.channel == 'XTION':
                    lcm_msg = rgbd_t.decode(event.data)
                    rgb_img = cv2.imdecode(np.fromstring(
                        lcm_msg.rgb, np.uint8), cv2.IMREAD_COLOR)
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                    if args.visualise:
                        cv2.imshow('RGB', rgb_img)
                        cv2.waitKey(10)

                    depth_img = np.fromstring(zlib.decompress(lcm_msg.depth), np.uint16).reshape(
                        lcm_msg.height, lcm_msg.width)

                    rgb_ros_msg = bridge.cv2_to_imgmsg(rgb_img, 'bgr8')
                    rgb_ros_msg.header.seq = event.eventnum

                    secs_float = float(lcm_msg.utime)/1e6
                    nsecs_float = (secs_float - np.floor(secs_float)) * 1e9
                    rgb_ros_msg.header.stamp.secs = int(secs_float)
                    rgb_ros_msg.header.stamp.nsecs = int(nsecs_float)
                    rgb_ros_msg.header.frame_id = args.namespace

                    depth_ros_msg = bridge.cv2_to_imgmsg(depth_img)
                    depth_ros_msg.header.seq = event.eventnum
                    depth_ros_msg.header = rgb_ros_msg.header

                    camera_info = CameraInfo()
                    camera_info.header = rgb_ros_msg.header
                    camera_info.height = rgb_img.shape[0]
                    camera_info.width = rgb_img.shape[1]

                    camera_info.distortion_model = 'plumb_bob'
                    camera_info.K = config["K"]

                    bag.write(args.namespace+'/rgb/image_raw',
                            rgb_ros_msg, event_stamp)
                    bag.write(args.namespace+'/depth/image_rect',
                            depth_ros_msg, event_stamp)
                    bag.write(args.namespace+'/camera_info',
                            camera_info, event_stamp)

                if event.channel == 'MICROSTRAIN_RAW':
                    lcm_msg = raw_t.decode(event.data)

                    imu_ros_msg = Imu()
                    secs_float = float(lcm_msg.utime)/1e9
                    nsecs_float = (secs_float - np.floor(secs_float)) * 1e9

                    imu_ros_msg.header.seq = event.eventnum
                    imu_ros_msg.header.stamp.secs = int(secs_float)
                    imu_ros_msg.header.stamp.nsecs = int(nsecs_float)
                    imu_ros_msg.header.frame_id = "imu"

                    imu_ros_msg.linear_acceleration.x = lcm_msg.accel[0]
                    imu_ros_msg.linear_acceleration.y = lcm_msg.accel[1]
                    imu_ros_msg.linear_acceleration.z = lcm_msg.accel[2]

                    imu_ros_msg.angular_velocity.x = lcm_msg.gyro[0]
                    imu_ros_msg.angular_velocity.y = lcm_msg.gyro[1]
                    imu_ros_msg.angular_velocity.z = lcm_msg.gyro[2]

                    # Store magnetometer in magnetometer msg
                    mag_msg = MagneticField()
                    mag_msg.header = imu_ros_msg.header
                    mag_msg.magnetic_field.x = lcm_msg.mag[0]
                    mag_msg.magnetic_field.y = lcm_msg.mag[1]
                    mag_msg.magnetic_field.z = lcm_msg.mag[2]
                    # Ignore pressure for now
                    # lcm_msg.pressure

                    bag.write('imu_raw', imu_ros_msg, event_stamp)
                    bag.write('imu_mag', mag_msg, event_stamp)
                
                bar.update(event.__sizeof__() + event.data.__sizeof__())

        

    finally:
        log.close()
        bag.close()

    print("Done.")

if __name__ == '__main__':
    main()
