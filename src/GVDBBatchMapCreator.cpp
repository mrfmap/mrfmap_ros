/**
 * Simple class to take in a list of images and poses and build a MRFMap
 **/

#include <codetimer/codetimer.h>
#include <mrfmap_ros/GVDBBatchMapCreator.h>

#include <chrono>

extern gvdb_params_t gvdb_params_;

std::string to_zero_lead(const int value, const unsigned precision) {
  std::ostringstream oss;
  oss << std::setw(precision) << std::setfill('0') << value;
  return oss.str();
}

GVDBBatchMapCreator::GVDBBatchMapCreator(std::string config_file) : cam_in_body_(Eigen::Matrix4f::Identity()), world_frame_transform_(Eigen::Matrix4f::Identity()), img_dt_(0.0f), depth_img_scale_(1.0f), camera_topic_(""), odom_topic_(""), selector_(0.5f, 0.5f), generate_octomap_(false), legacy_odom_msg_(false), icl_bag_(false) {
  read_config(config_file);
  init_subscribers();
  inf_ = std::make_shared<GVDBInference>(true, false);
  if (generate_octomap_) {
    octomap_ = std::make_shared<GVDBOctomapWrapper>(gvdb_params_);
  }
}

void GVDBBatchMapCreator::init_subscribers() {
  ros::Time::init();
  image_sub_ = std::make_shared<AddLagSubscriber<sensor_msgs::Image>>(img_dt_);

  if (legacy_odom_msg_) {
    odom_sub_ = std::make_shared<AddLagSubscriber<nav_msgs::Odometry>>(0.0);
    odom_synchronizer_ = std::make_shared<message_filters::Synchronizer<MySyncPolicyOdom>>(MySyncPolicyOdom(10), *image_sub_, *odom_sub_);
    odom_synchronizer_->registerCallback(
        boost::bind(&GVDBBatchMapCreator::callback_odom, this, _1, _2));
  } else {
    pose_sub_ = std::make_shared<AddLagSubscriber<geometry_msgs::PoseStamped>>(0.0);
    pose_synchronizer_ = std::make_shared<message_filters::Synchronizer<MySyncPolicyPose>>(MySyncPolicyPose(10), *image_sub_, *pose_sub_);
    pose_synchronizer_->registerCallback(
        boost::bind(&GVDBBatchMapCreator::callback_pose, this, _1, _2));
  }
}

void GVDBBatchMapCreator::add_keyframe(const Eigen::MatrixXf &pose, const Eigen::MatrixXf &image) {
  inf_->add_camera_with_depth(pose, image);
  inf_->perform_inference();
  if (generate_octomap_) {
    LOG(std::cout << "...Adding octomap image\n");
    auto start_block1 = std::chrono::high_resolution_clock::now();
    octomap_->add_camera_with_depth(pose, image);
    CodeTimer::record("octomap" + to_zero_lead(octomap_->get_num_cams(), 3), start_block1);
    LOG(std::cout << "...done!\n");
  }
}

void GVDBBatchMapCreator::callback_pose(const sensor_msgs::Image::ConstPtr &image_msg,
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
  // Need this for the Stanford bag file
  if (icl_bag_) {
    cam_in_world = world_frame_transform_ * pose * cam_in_body_;
  } else {
    cam_in_world = pose * cam_in_body_;
  }
  if (selector_.is_keyframe(pose)) {
    images_.push_back(eigen_img);
    poses_.push_back(cam_in_world);
  }
}

void GVDBBatchMapCreator::callback_odom(const sensor_msgs::Image::ConstPtr &image_msg,
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
    images_.push_back(eigen_img);
    poses_.push_back(cam_in_world);
  }
}

void GVDBBatchMapCreator::process_saved_tuples() {
  for (int i = 0; i < poses_.size(); ++i) {
    add_keyframe(poses_[i], images_[i]);
  }
  // Also populate Likelihood GVDB volume of octomapwrapper
  if (generate_octomap_) {
    LOG(std::cout << "[GVDBBatchMapCreator] About to push octomap to its gvdb volume...\n");
    octomap_->push_to_gvdb_volume();
    LOG(std::cout << "[GVDBBatchMapCreator] Done!\n");
  }
  LOG(std::cout << "[GVDBBatchMapCreator] Done creating batch map!\n");
  CodeTimer::printStats();
}

std::string GVDBBatchMapCreator::get_stats() {
  return CodeTimer::streamStats().str();
}

void GVDBBatchMapCreator::read_config(std::string params_file) {
  try {
    std::cout << "Trying to read file " << params_file << "\n";
    YAML::Node params = YAML::LoadFile(params_file);
    YAML::Node rosviewer_params_node = params["viewer_params_nodes"];
    if (!rosviewer_params_node) {
      std::cerr
          << "[GVDBBatchMapCreator] Could not read viewer_params_nodes!";
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
      YAML::Node camera_topic_node = rosviewer_params_node["camera_topic"];
      if (camera_topic_node) {
        camera_topic_ = camera_topic_node.as<std::string>();
      }

      YAML::Node odom_topic_node = rosviewer_params_node["odom_topic"];
      if (odom_topic_node) {
        odom_topic_ = odom_topic_node.as<std::string>();
      }

      YAML::Node legacy_mode_node = rosviewer_params_node["is_legacy_odom_msg"];
      if (legacy_mode_node) {
        legacy_odom_msg_ = legacy_mode_node.as<bool>();
      }

      YAML::Node icl_bag_node = rosviewer_params_node["is_icl_bag"];
      if (icl_bag_node) {
        icl_bag_ = icl_bag_node.as<bool>();
      }

      YAML::Node img_scale_node = rosviewer_params_node["img_scale_factor"];
      if (img_scale_node) {
        depth_img_scale_ = img_scale_node.as<float>();
      }

      YAML::Node img_dt_node = rosviewer_params_node["img_lag"];
      if (img_dt_node) {
        img_dt_ = img_dt_node.as<float>();
      }

      YAML::Node thresh_node_ = rosviewer_params_node["translation_thresh"];
      if (thresh_node_) {
        float trans_thresh = thresh_node_.as<float>();
        float rot_thresh = rosviewer_params_node["rotation_thresh"].as<float>();
        selector_.set_thresh(rot_thresh, trans_thresh);
      }

      YAML::Node octo_toggle_node = rosviewer_params_node["view_octomap"];
      if (octo_toggle_node) {
        generate_octomap_ = octo_toggle_node.as<bool>();
      }

      std::cout << "[GVDBBatchMapCreator] Loaded \n\tcam_in_body : \n"
                << cam_in_body_ << "\n\tCamera topic:: " << camera_topic_ << "\n\tOdom topic:: " << odom_topic_ << "\nIs legacy odom:: " << legacy_odom_msg_ << "\n\tDepth scale:: " << depth_img_scale_ << "\n\tImage temporal lag:: " << img_dt_ << "\n";
    }
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    exit(-1);
  }
}

// Load bag
void GVDBBatchMapCreator::load_bag(const std::string &filename) {
  rosbag::Bag bag;
  bag.open(filename, rosbag::bagmode::Read);

  // Image topics to load
  std::vector<std::string> topics;
  topics.push_back(camera_topic_);
  topics.push_back(odom_topic_);

  rosbag::View view(bag, rosbag::TopicQuery(topics));
  std::cout << "\n";
  for (rosbag::MessageInstance const m : view) {
    if (m.getTopic() == camera_topic_ || ("/" + m.getTopic() == camera_topic_)) {
      sensor_msgs::Image::ConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
      if (img_msg != NULL)
        image_sub_->newMessage(img_msg);
    }

    if (m.getTopic() == odom_topic_ || ("/" + m.getTopic() == odom_topic_)) {
      if (legacy_odom_msg_) {
        nav_msgs::Odometry::ConstPtr odom_msg = m.instantiate<nav_msgs::Odometry>();
        if (odom_msg != NULL)
          odom_sub_->newMessage(odom_msg);
      } else {
        geometry_msgs::PoseStamped::ConstPtr pose_msg = m.instantiate<geometry_msgs::PoseStamped>();
        if (pose_msg != NULL)
          pose_sub_->newMessage(pose_msg);
      }
    }

    printf("\r Read bag ration:: %f\t", ((m.getTime() - view.getBeginTime()).toSec() / (view.getEndTime() - view.getBeginTime()).toSec()));
  }
  bag.close();
}