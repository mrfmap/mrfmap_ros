#pragma once

#include <cuda_runtime_api.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
#include <message_filters/simple_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <mrfmap/GVDBInference.h>
#include <mrfmap/GVDBOctomapWrapper.h>
#include <mrfmap/KeyframeSelector.h>
#include <mrfmap/Viewer.h>
#include <nav_msgs/Odometry.h>
#include <ros/node_handle.h>
#include <ros/subscriber.h>
#include <sensor_msgs/Image.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

/**
 * Inherits from message_filters::SimpleFilter<M>
 * to use protected signalMessage function.
 * Takes incoming message and removes a constant reql-world delay from its timestamp
 * to work with approxtimesynchroniser
 */
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

class RosViewerWrapper {
 private:
  std::shared_ptr<ros::NodeHandle> nh_;
  std::shared_ptr<image_transport::ImageTransport> it_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> image_sub_;
  std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;
  std::shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
  std::shared_ptr<AddLagSubscriber<sensor_msgs::Image>> delayed_image_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
      MySyncPolicyOdom;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
      MySyncPolicyPose;
  std::shared_ptr<message_filters::Synchronizer<MySyncPolicyOdom>> odom_synchronizer_;
  std::shared_ptr<message_filters::Synchronizer<MySyncPolicyPose>> pose_synchronizer_;

  KeyframeSelector selector_;
  std::shared_ptr<PangolinViewer> viewer_;
  Eigen::MatrixXf cam_in_body_, world_frame_transform_;
  std::string camera_topic_, odom_topic_;
  bool mrfmap_only_mode_, render_likelihoods_, legacy_odom_msg_, icl_bag_;
  float depth_img_scale_;
  float img_dt_;

 public:
  RosViewerWrapper(ros::NodeHandle nh, std::string params_file) : selector_(0.5f, 0.5f), cam_in_body_(Eigen::Matrix4f::Identity()), world_frame_transform_(Eigen::Matrix4f::Identity()), camera_topic_("/camera/depth/image_raw"), odom_topic_("/camera/vicon_odom"), depth_img_scale_(1.0f), mrfmap_only_mode_(true), render_likelihoods_(true), legacy_odom_msg_(false), icl_bag_(false), img_dt_(0.0f) {
    nh_ = std::make_shared<ros::NodeHandle>(nh);
    it_ = std::make_shared<image_transport::ImageTransport>(*nh_);
    read_config(params_file);
    viewer_ = std::make_shared<PangolinViewer>(std::string("MRFMap Viewer"), mrfmap_only_mode_, render_likelihoods_);
    init_subscribers();
    // Register exit callback
    viewer_->register_exit_callback(std::bind(&RosViewerWrapper::exit_callback, this));
  }

  void exit_callback() {
    std::cout << "In exit function!\n";
    ros::requestShutdown();
  }

  void init_subscribers() {
    image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(*nh_, camera_topic_, 100);
    delayed_image_ =
        std::make_shared<AddLagSubscriber<sensor_msgs::Image>>(img_dt_);
    // Send the image data callback to our custom filter to incorporate calibrated temporal lag
    image_sub_->registerCallback(
        boost::bind(&AddLagSubscriber<sensor_msgs::Image>::newMessage, delayed_image_, _1));

    if (legacy_odom_msg_) {
      odom_sub_ = std::make_shared<message_filters::Subscriber<nav_msgs::Odometry>>(*nh_, odom_topic_, 100);
      odom_synchronizer_ = std::make_shared<message_filters::Synchronizer<MySyncPolicyOdom>>(MySyncPolicyOdom(100), *delayed_image_, *odom_sub_);
      odom_synchronizer_->registerCallback(
          boost::bind(&RosViewerWrapper::callback_odom, this, _1, _2));
    } else {
      pose_sub_ = std::make_shared<message_filters::Subscriber<geometry_msgs::PoseStamped>>(*nh_, odom_topic_, 100);
      pose_synchronizer_ = std::make_shared<message_filters::Synchronizer<MySyncPolicyPose>>(MySyncPolicyPose(100), *delayed_image_, *pose_sub_);
      pose_synchronizer_->registerCallback(
          boost::bind(&RosViewerWrapper::callback_pose, this, _1, _2));
    }
  }

  void callback_odom(const sensor_msgs::Image::ConstPtr &image_msg,
                     const nav_msgs::Odometry::ConstPtr &odom_msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception &e) {
      throw std::runtime_error(
          std::string("cv_bridge exception: ") + std::string(e.what()));
    }
    Eigen::MatrixXf eigen_img = Eigen::MatrixXf::Zero(cv_ptr->image.rows, cv_ptr->image.cols);
    cv::cv2eigen(cv_ptr->image, eigen_img);

    // TODO: Check if we need to scale depth image depending on sensor image encoding...
    eigen_img *= depth_img_scale_;

    Eigen::MatrixXf pose = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf quat(odom_msg->pose.pose.orientation.w,
                            odom_msg->pose.pose.orientation.x,
                            odom_msg->pose.pose.orientation.y,
                            odom_msg->pose.pose.orientation.z);
    Eigen::Vector3f pos(odom_msg->pose.pose.position.x,
                        odom_msg->pose.pose.position.y,
                        odom_msg->pose.pose.position.z);
    pose.block<3, 3>(0, 0) = quat.toRotationMatrix();
    pose.block<3, 1>(0, 3) = pos;

    // Transform the pose by cam_in_body
    Eigen::MatrixXf cam_in_world = Eigen::Matrix4f::Identity();
    // Need this for the ICL bag file
    if (icl_bag_) {
      cam_in_world = world_frame_transform_ * pose * cam_in_body_;
    } else {
      cam_in_world = pose * cam_in_body_;
    }
    if (selector_.is_keyframe(pose)) {
      viewer_->add_keyframe(cam_in_world, eigen_img);
    } else {
      viewer_->add_frame(cam_in_world, eigen_img);
    }
  }

  void callback_pose(const sensor_msgs::Image::ConstPtr &image_msg,
                     const geometry_msgs::PoseStamped::ConstPtr &odom_msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception &e) {
      throw std::runtime_error(
          std::string("cv_bridge exception: ") + std::string(e.what()));
    }
    Eigen::MatrixXf eigen_img = Eigen::MatrixXf::Zero(cv_ptr->image.rows, cv_ptr->image.cols);
    cv::cv2eigen(cv_ptr->image, eigen_img);

    // TODO: Check if we need to scale depth image depending on sensor image encoding...
    eigen_img *= depth_img_scale_;

    Eigen::MatrixXf pose = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf quat(odom_msg->pose.orientation.w,
                            odom_msg->pose.orientation.x,
                            odom_msg->pose.orientation.y,
                            odom_msg->pose.orientation.z);
    Eigen::Vector3f pos(odom_msg->pose.position.x,
                        odom_msg->pose.position.y,
                        odom_msg->pose.position.z);

    pose.block<3, 3>(0, 0) = quat.toRotationMatrix();
    pose.block<3, 1>(0, 3) = pos;

    // Transform the pose by cam_in_body
    Eigen::MatrixXf cam_in_world = Eigen::Matrix4f::Identity();
    // Need this for the ICL bag file
    if (icl_bag_) {
      cam_in_world = world_frame_transform_ * pose * cam_in_body_;
    } else {
      cam_in_world = pose * cam_in_body_;
    }

    if (selector_.is_keyframe(pose)) {
      viewer_->add_keyframe(cam_in_world, eigen_img);
    } else {
      viewer_->add_frame(cam_in_world, eigen_img);
    }
  }

  void read_config(std::string params_file) {
    try {
      std::cout << "Trying to read file " << params_file << "\n";
      YAML::Node params = YAML::LoadFile(params_file);
      YAML::Node rosviewer_params_node = params["viewer_params_nodes"];
      if (!rosviewer_params_node) {
        std::cerr
            << "[RosMRFMapNode] Could not read viewer_params_nodes!";
        exit(-1);
      } else {
        gvdb_params_.load_from_file(params_file);
        YAML::Node cam_in_body = rosviewer_params_node["cam_in_body"];
        if (cam_in_body) {
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              cam_in_body_(i, j) = cam_in_body[i * 4 + j].as<float>();
            }
          }
        }

        YAML::Node world_frame_transform = rosviewer_params_node["world_frame_transform"];
        if (world_frame_transform) {
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              world_frame_transform_(i, j) = world_frame_transform[i * 4 + j].as<float>();
            }
          }
        }

        YAML::Node legacy_mode_node = rosviewer_params_node["is_legacy_odom_msg"];
        if (legacy_mode_node) {
          legacy_odom_msg_ = legacy_mode_node.as<bool>();
        }

        YAML::Node icl_bag_node = rosviewer_params_node["is_icl_bag"];
        if (icl_bag_node) {
          icl_bag_ = icl_bag_node.as<bool>();
        }

        YAML::Node camera_topic_node = rosviewer_params_node["camera_topic"];
        if (camera_topic_node) {
          camera_topic_ = camera_topic_node.as<std::string>();
        }

        YAML::Node odom_topic_node = rosviewer_params_node["odom_topic"];
        if (odom_topic_node) {
          odom_topic_ = odom_topic_node.as<std::string>();
        }

        YAML::Node img_scale_node = rosviewer_params_node["img_scale_factor"];
        if (img_scale_node) {
          depth_img_scale_ = img_scale_node.as<float>();
        }

        YAML::Node img_dt_node = rosviewer_params_node["img_lag"];
        if (img_dt_node) {
          img_dt_ = img_dt_node.as<float>();
        }

        YAML::Node view_octo_node = rosviewer_params_node["view_octomap"];
        if (view_octo_node) {
          mrfmap_only_mode_ = !view_octo_node.as<bool>();
        }

        YAML::Node render_likelihoods_node = rosviewer_params_node["render_likelihoods"];
        if (render_likelihoods_node) {
          render_likelihoods_ = render_likelihoods_node.as<bool>();
        }

        YAML::Node thresh_node_ = rosviewer_params_node["translation_thresh"];
        if (thresh_node_) {
          float trans_thresh = thresh_node_.as<float>();
          float rot_thresh = rosviewer_params_node["rotation_thresh"].as<float>();
          selector_.set_thresh(rot_thresh, trans_thresh);
        }

        std::cout << "[RosMRFMapNode] Loaded cam_in_body : \n"
                  << cam_in_body_ << "\n Camera topic:: " << camera_topic_ << "\n Odom topic:: " << odom_topic_ << "\n Depth scale:: " << depth_img_scale_ << "\n Image temporal lag:: " << img_dt_ << "\n Is ICL bag:: " << icl_bag_ << "\n";
      }
    } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      exit(-1);
    }
  }
};