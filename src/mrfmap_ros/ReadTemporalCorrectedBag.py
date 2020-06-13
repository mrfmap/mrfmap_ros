import numpy as np
import rospy
import tf
import rosbag
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import cv_bridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import click
import cv2

dtype = np.float32

class ReadTemporalCorrectedBag:
    def __init__(self, topics_to_parse, path, offset_dts, scale_factors, world_frame_transform):
        self.topics_to_parse = topics_to_parse
        self.duplicate = False
        if len(topics_to_parse) == 2 or topics_to_parse[2] == topics_to_parse[0]:
            # We use the same data stream for building and eval
            print 'Duplicate image streams! Not running another synchroniser!'
            self.duplicate = True
        self.path = path
        self.offset_dts = offset_dts
        self.scale_factors = scale_factors
        self.rs_data = []
        self.kin_data = []
        self.subs = []
        self.img_sub = Subscriber(topics_to_parse[0], Image)
        self.subs.append(self.img_sub)
        self.pose_sub = Subscriber(topics_to_parse[1], PoseStamped)
        self.subs.append(self.pose_sub)
        self.world_frame_transform = world_frame_transform

        if not self.duplicate:
            if topics_to_parse[2] != topics_to_parse[0]:
                self.subs.append(Subscriber(topics_to_parse[2], Image))
            else:
                self.subs.append(self.img_sub)
            if topics_to_parse[3] != topics_to_parse[1]:
                self.subs.append(Subscriber(topics_to_parse[3], PoseStamped))
            else:
                self.subs.append(self.pose_sub)
        self.rs_synchronizer = ApproximateTimeSynchronizer([self.subs[0], self.subs[1]], 100, 0.05)
        self.rs_synchronizer.registerCallback(self.got_tuple, 'rs')
        if not self.duplicate:
            self.kin_synchronizer = ApproximateTimeSynchronizer([self.subs[2], self.subs[3]], 100, 0.05)
            self.kin_synchronizer.registerCallback(self.got_tuple, 'kin')
        self.bridge = cv_bridge.CvBridge()
        print 'Topics to parse are '+str(topics_to_parse)

    def read(self):
        rospy.init_node('icra_realworld')
        with rosbag.Bag(self.path, 'r') as bag:
            bag_length = bag.get_message_count(self.topics_to_parse)
            with click.progressbar(bag.read_messages(self.topics_to_parse),
                                   length=bag_length) as bar:
                for topic, msg, stamp in bar:
                    if topic in self.topics_to_parse:
                        index = self.topics_to_parse.index(topic)
                        if index in [1, 3]:
                            # Check if we have a legacy vicon Odometry message
                            if str(msg._type) == 'nav_msgs/Odometry':
                                new_msg = PoseStamped()
                                new_msg.pose = msg.pose.pose
                                new_msg.header = msg.header
                                msg = new_msg
                        else:
                            # Adjust dt for mocap to value obtained from lag_detector
                            if index == 0:
                                dt = self.offset_dts[0]
                            elif index == 2:
                                dt = self.offset_dts[1]
                            msg.header.stamp = rospy.Time(
                                msg.header.stamp.to_sec() - dt)
                        self.subs[index].signalMessage(msg)
        if self.duplicate:
            self.kin_data = self.rs_data

    def transform_matrix_from_odom(self, msg):
        translation = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        quaternion = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                               msg.pose.orientation.z, msg.pose.orientation.w])
        T = tf.transformations.quaternion_matrix(quaternion)
        T[:3, 3] = translation
        return T.astype(dtype)

    def got_tuple(self, img_msg, pose_msg, type_str):
        img = self.bridge.imgmsg_to_cv2(img_msg, "passthrough").astype(dtype)
        pose = self.transform_matrix_from_odom(pose_msg)
        # Need to add world transform
        pose = np.dot(self.world_frame_transform, pose)
        if type_str == 'rs':
            self.rs_data.append([img*self.scale_factors[0], pose])
        else:
            self.kin_data.append([img*self.scale_factors[1], pose])

