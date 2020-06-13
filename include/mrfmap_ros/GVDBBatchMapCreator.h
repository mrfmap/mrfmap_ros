#pragma once

#include <cuda_runtime_api.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <message_filters/simple_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>

#include <mrfmap/GVDBInference.h>
#include <mrfmap/KeyframeSelector.h>

#include <mrfmap/GVDBOctomapWrapper.h>
#include <yaml-cpp/yaml.h>

template <class M>
class AddLagSubscriber : public message_filters::SimpleFilter<M> {
 public:
  AddLagSubscriber(float dt) : dt_(dt){};
  void newMessage(const boost::shared_ptr<M const> &msg) {
    // Make a copy of incoming pointer and change the header stamp attribute
    M copy(*msg);
    copy.header.stamp = msg->header.stamp - ros::Duration(dt_);
    const boost::shared_ptr<M const> copy_ptr = boost::make_shared<M const>(copy);
    this->signalMessage(copy_ptr);
  }

 private:
  float dt_;
};

class GVDBBatchMapCreator {
 private:
  std::shared_ptr<GVDBInference> inf_;
  KeyframeSelector selector_;

  Eigen::MatrixXf cam_in_body_, world_frame_transform_;
  std::string camera_topic_, odom_topic_;
  float depth_img_scale_;
  float img_dt_;

  bool generate_octomap_, legacy_odom_msg_, icl_bag_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
      MySyncPolicyPose;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
      MySyncPolicyOdom;
  std::shared_ptr<message_filters::Synchronizer<MySyncPolicyPose>> pose_synchronizer_;
  std::shared_ptr<message_filters::Synchronizer<MySyncPolicyOdom>> odom_synchronizer_;

  std::shared_ptr<AddLagSubscriber<sensor_msgs::Image>> image_sub_;
  std::shared_ptr<AddLagSubscriber<geometry_msgs::PoseStamped>> pose_sub_;
  std::shared_ptr<AddLagSubscriber<nav_msgs::Odometry>> odom_sub_;

  std::vector<Eigen::MatrixXf> images_;
  std::vector<Eigen::MatrixXf> poses_;

  std::shared_ptr<GVDBOctomapWrapper> octomap_;

 public:
  GVDBBatchMapCreator(std::string config_file);
  ~GVDBBatchMapCreator() {
    DEBUG(std::cout<<"[GVDBBatchMapCreator] Deleting object!\n");
    pose_synchronizer_.reset();
    image_sub_.reset();
    pose_sub_.reset();
    odom_sub_.reset();
    images_.clear();
    poses_.clear();
    inf_.reset();
    if (generate_octomap_) {
      octomap_.reset();
    }
  }
  int num_tuples() {
    return poses_.size();
  }

  void init_subscribers();
  void add_keyframe(const Eigen::MatrixXf &pose, const Eigen::MatrixXf &image);
  void callback_pose(const sensor_msgs::Image::ConstPtr &image_msg,
                     const geometry_msgs::PoseStamped::ConstPtr &odom_msg);
  void callback_odom(const sensor_msgs::Image::ConstPtr &image_msg,
                     const nav_msgs::Odometry::ConstPtr &odom_msg);
  void process_saved_tuples();
  std::string get_stats();
  void read_config(std::string params_file);
  void load_bag(const std::string &filename);
  std::shared_ptr<GVDBInference> get_inference_ptr() {
    return inf_;
  }
  std::shared_ptr<GVDBMapLikelihoodEstimator> get_octomap_ptr() {
    return octomap_->octo_;
  }
};