#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
import cv2
from gazebo_msgs.msg import LinkStates, LinkState
from gazebo_msgs.srv import SetLinkState, GetLinkState
from std_srvs.srv import Empty
import tf
import pdb

# plt.ion()


class GazeboOrbitDataCollector:

    def __init__(self, width, height, resolution=0.05, debug=False):
        self.bridge = CvBridge()
        self.latest_pose = None
        self.latest_depth_img = None
        self.latest_rgb_img = None
        self.rgb_img = None
        self.num = 0
        self.init_subscribers()

        self.debug_images = []

    def init_subscribers(self):
        self.get_link_service = rospy.ServiceProxy(
            '/gazebo/get_link_state', GetLinkState)
        self.depth_image_sub = rospy.Subscriber(
            '/camera/depth/image_raw', Image, self.depth_img_callback)
        self.image_sub = rospy.Subscriber(
            '/camera/rgb/image_raw', Image, self.rgb_img_callback)
#         self.link_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_callback)

    def link_callback(self, msg):
        index = msg.name.index('simple_rgbd_line_cam::link1')
        pos = msg.pose[index].position
        quat = msg.pose[index].orientation

        q = np.array([quat.x, quat.y, quat.z, quat.w])
        p = np.array([pos.x, pos.y, pos.z])

        latest_pose = tf.transformations.quaternion_matrix(q)
        latest_pose[:3, 3] = p
        self.latest_pose = latest_pose.copy()

    def rgb_img_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as e:
            print(e)

        self.latest_rgb_img = cv_image.copy()

    def depth_img_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as e:
            print(e)

        self.latest_depth_img = cv_image.copy()
        response = self.get_link_service('simple_rgbd_line_cam::link1', '')
        if response.success:
            pos = response.link_state.pose.position
            quat = response.link_state.pose.orientation

            q = np.array([quat.x, quat.y, quat.z, quat.w])
            p = np.array([pos.x, pos.y, pos.z])

            latest_pose = tf.transformations.quaternion_matrix(q)
            latest_pose[:3, 3] = p
            self.latest_pose = latest_pose.copy()
            self.rgb_img = self.latest_rgb_img.copy()

    def save_to_binary(self, path, data):
        with open(path, 'wb') as outfile:
            outfile.write(bytearray(np.array(data.shape).astype('uint64')))
            outfile.write(bytearray(data.flatten('F')))

    def update(self):
        if self.latest_depth_img is not None and self.latest_pose is not None:
            np.save('depth_'+str(self.num), self.latest_depth_img)
            np.save('pose_'+str(self.num), self.latest_pose)
            np.save('rgb_'+str(self.num), self.rgb_img)
            self.save_to_binary('depth_'+str(self.num)+'.bin',
                                self.latest_depth_img.astype(np.float32))
            self.save_to_binary('pose_'+str(self.num)+'.bin',
                                self.latest_pose.astype(np.float32))
            self.num += 1
            self.latest_depth_img = None
            return True
        else:
            print 'Data not ready'
            return False


class OrbitTrajectory:

    def __init__(self, radius=3.0, time_period=10.0):
        self.radius = radius
        self.orient_inwards = True
        self.elevation = 2.0
        self.time_period = time_period

    def get_current_pose(self, t):
        '''
        Return the pose corresponding to the current time
        '''
        frac = t / self.time_period
        x = self.radius * np.cos(2.0 * np.pi * frac)
        y = self.radius * np.sin(2.0 * np.pi * frac)
        z = self.elevation

        yaw = 2.0 * np.pi * frac
        if self.orient_inwards:
            yaw += np.pi

        pitch = np.pi/6

        print 'outgoing yaw is '+str(yaw)
        pose = tf.transformations.euler_matrix(0, pitch, yaw)
        pose[:3, 3] = np.array([x, y, z])

        return pose


if __name__ == '__main__':
    rospy.init_node('occupancy_node')
    width = 6
    height = 6
    resolution = 0.05
    debug = True
    wrapper = GazeboOrbitDataCollector(width, height, resolution, debug)
    rate = rospy.Rate(1)
    radius = 3.0
    time_period = 12.0
    t = 0
    traj = OrbitTrajectory(radius, time_period)

    print 'Waiting for Gazebo. Start the roslaunch?'
    rospy.wait_for_service('/gazebo/set_link_state')
    set_link_service = rospy.ServiceProxy(
        '/gazebo/set_link_state', SetLinkState)
    rospy.wait_for_service('/gazebo/set_physics_properties')
    pause_physics_service = rospy.ServiceProxy(
        '/gazebo/pause_physics', Empty)
    pause_physics_service()
    state = LinkState()
    state.pose.position.x = -3
    state.link_name = 'simple_rgbd_line_cam::link1'
    state.reference_frame = 'world'
    state.pose.orientation.w = 1

    xs = []
    ys = []
    while not rospy.is_shutdown():
        try:
            pose = traj.get_current_pose(t)
            pos = pose[:3, 3]
            quat = tf.transformations.quaternion_from_matrix(pose)

            state.pose.position.x = pos[0]
            state.pose.position.y = pos[1]
            state.pose.position.z = pos[2]

            state.pose.orientation.w = quat[3]
            state.pose.orientation.x = quat[0]
            state.pose.orientation.y = quat[1]
            state.pose.orientation.z = quat[2]

            response = set_link_service(state)
            xs.append(pos[0])
            ys.append(pos[1])
            print response, state.pose.position
            rospy.sleep(1)
            wrapper.update()
            print t
            t += 1
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        rate.sleep()
        if t > time_period:
            break
