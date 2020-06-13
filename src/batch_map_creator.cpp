#include <mrfmap_ros/GVDBBatchMapCreator.h>

gvdb_params_t gvdb_params_;

int main(int argc, char const *argv[]) {
  std::string config_file = "../config/realsense_848.yaml";
  std::string bag_file;
  ros::Time::init();
  if (argc > 2) {
    config_file = argv[1];
    bag_file = argv[2];
  } else {
    bag_file = argv[1];
  }
  {
    std::cout << "Reading config file :: " << config_file << "\n";
    GVDBBatchMapCreator creator(config_file);
    std::cout << "Reading bag file :: " << bag_file << "\n";
    creator.load_bag(bag_file);
    std::cout << creator.num_tuples() << " tuples read. Starting inference.\n";
    creator.process_saved_tuples();
    std::cout << "Done.\n";
  }
  return 1;
}