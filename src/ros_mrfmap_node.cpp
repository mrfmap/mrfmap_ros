#include <mrfmap_ros/RosViewerWrapper.h>

gvdb_params_t gvdb_params_;

int main(int argc, char** argv) {
  ros::init(argc, argv, "ros_mrfmap_node");
  ros::NodeHandle nh("~");
  std::string config_file;
  nh.param<std::string>("config_file", config_file, "../config/realsense.yaml");

  RosViewerWrapper wrapper(nh, config_file);
  ros::Rate r(60);
  do {
    ros::spinOnce();
    r.sleep();
  } while (ros::ok());
  std::cout << "Toodles!\n";
  return EXIT_SUCCESS;
}
