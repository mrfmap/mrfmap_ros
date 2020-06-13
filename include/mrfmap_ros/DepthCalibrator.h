#pragma once
#include <mrfmap/GVDBParams.h>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

extern gvdb_params_t gvdb_params_;

class DepthCalibrator {
 private:
  int cell_width_, cell_height_;
  int stride_;
  float z_min_, z_max_, z_bin_stride_;
  int z_bin_count_max_;
  std::vector<std::vector<float>> data_;
  std::vector<std::vector<int>> z_bin_counts_;
  std::vector<std::vector<std::vector<size_t>>> z_bin_indices_;
  std::mutex mutex_;

 public:
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> MRowf;
  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> MRowi;
  typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> MRowidx;
  DepthCalibrator(int cell_width, int cell_height, float z_min, float z_max, float z_bin_stride, int z_bin_count_max)
      : cell_width_(cell_width),
        cell_height_(cell_height),
        z_min_(z_min),
        z_max_(z_max),
        z_bin_stride_(z_bin_stride),
        z_bin_count_max_(z_bin_count_max) {
    stride_ = static_cast<int>(ceilf(1.0f * gvdb_params_.cols / cell_width_));
    int num_z_bins = static_cast<int>(ceilf((z_max_ - z_min_) / z_bin_stride_));
    // Initialize map
    size_t dim = static_cast<int>(ceilf(1.0f * gvdb_params_.rows / cell_height_)) * stride_;
    std::cout << "Image Dims::" << dim << " Number of Z depth bins::" << num_z_bins << "\n";
    data_.resize(dim);
    z_bin_counts_.resize(num_z_bins);
    z_bin_indices_.resize(num_z_bins);
    for (int i = 0; i < num_z_bins; ++i) {
      // We know how many bins we can actually store
      z_bin_counts_[i].resize(dim, 0);
      z_bin_indices_[i].resize(dim, std::vector<size_t>());
    }
  }

  bool AddImage(Eigen::Ref<MRowf>, const Eigen::MatrixXf &);
  size_t GetBinSize(int, int);
  int GetZBinSize(int, int, int);
  void GetBinData(int, int, Eigen::Ref<MRowf>);
  void GetZBinData(int, int, int, Eigen::Ref<MRowf>);
  void GetZBinCounts(int, Eigen::Ref<DepthCalibrator::MRowi>);
  void GetZBinIndices(int, int, int, Eigen::Ref<DepthCalibrator::MRowidx>);
  void FitBin(int);
  void SaveData(const std::string &);
  void ReadData(const std::string &);
};

bool DepthCalibrator::AddImage(Eigen::Ref<DepthCalibrator::MRowf> img, const Eigen::MatrixXf &gt_pose) {
  bool return_val = true;
  std::unique_lock<std::mutex> lock(mutex_);
  for (int i = 0; i < gvdb_params_.rows; ++i) {
    for (int j = 0; j < gvdb_params_.cols; ++j) {
      float measured_depth = img(i, j);
      if (measured_depth == 0.0f) {
        continue;
      } else {
        // Get camera ray - plane intersection
        // https://stackoverflow.com/questions/23975555/how-to-do-ray-plane-intersection
        Eigen::Vector3f cam_ray = {(j - gvdb_params_.K(0, 2)) / gvdb_params_.K(0, 0),
                                   (i - gvdb_params_.K(1, 2)) / gvdb_params_.K(1, 1),
                                   1.0f};
        cam_ray.normalize();
        Eigen::Vector3f normal = gt_pose.block<3, 1>(0, 2);
        float denom = cam_ray.dot(normal);
        if (denom > 0.01f) {
          float t = gt_pose.block<3, 1>(0, 3).dot(normal) / denom;
          if (t > 0) {
            float ground_truth_depth = cam_ray[2] * t;
            if (ground_truth_depth > z_max_ || fabsf(ground_truth_depth - measured_depth) > 1.0f * ground_truth_depth) {
              // std::cout << "\n WHOA. i,j::" << i << "," << j << " cam_ray::" << cam_ray.transpose() << " normal::" << normal.transpose() << " t::" << t << " measured_depth::" << measured_depth << " gt_pose::\n"
              // << gt_pose << "\n";
              return_val = false;
              continue;
            }

            uint u = j / cell_width_;
            uint v = i / cell_height_;

            int z_bin = static_cast<int>((ground_truth_depth - z_min_) / z_bin_stride_);
            // std::cout << "z_bin::" << z_bin << " (ground_truth_depth = " << ground_truth_depth << "measured_depth = " << measured_depth << "z_bin_stride = " << z_bin_stride_ << "\n";

            if (z_bin < z_bin_counts_.size() && z_bin_counts_[z_bin][v * stride_ + u] < z_bin_count_max_) {
              // Only add to data if this bin is still not saturated
              // This ensures that we don't bias the fit.
              data_[v * stride_ + u].push_back(measured_depth);
              data_[v * stride_ + u].push_back(ground_truth_depth);
              z_bin_indices_[z_bin][v * stride_ + u].push_back(data_[v * stride_ + u].size() - 2);
              ++z_bin_counts_[z_bin][v * stride_ + u];
            }
          }
        }
      }
    }
  }
  return return_val;
}

size_t DepthCalibrator::GetBinSize(int v, int u) {
  if (v * stride_ + u > data_.size()) {
    throw std::runtime_error("Requested bin is out of bounds!");
  }
  std::unique_lock<std::mutex> lock(mutex_);
  return data_[v * stride_ + u].size();
}

int DepthCalibrator::GetZBinSize(int v, int u, int d) {
  if (d > z_bin_counts_.size()) {
    throw std::runtime_error("Requested bin is out of bounds!");
  }
  std::unique_lock<std::mutex> lock(mutex_);
  return z_bin_counts_[d][v * stride_ + u];
}

void DepthCalibrator::GetBinData(int v, int u, Eigen::Ref<DepthCalibrator::MRowf> bin_data) {
  // Just cast the std::vector to the Eigen matrix
  std::unique_lock<std::mutex> lock(mutex_);
  bin_data = Eigen::Map<DepthCalibrator::MRowf>(data_[v * stride_ + u].data(), data_[v * stride_ + u].size(), 1);
}

void DepthCalibrator::GetZBinCounts(int w, Eigen::Ref<DepthCalibrator::MRowi> z_bin_counts) {
  if (w > z_bin_counts_[0].size() || w < 0) {
    throw std::runtime_error("Invalid ZBin index!");
  }
  // Just cast the std::vector to the Eigen matrix
  std::unique_lock<std::mutex> lock(mutex_);
  z_bin_counts = Eigen::Map<DepthCalibrator::MRowi>(z_bin_counts_[w].data(), z_bin_counts_[w].size(), 1);
}

void DepthCalibrator::GetZBinData(int v, int u, int w, Eigen::Ref<DepthCalibrator::MRowf> z_bin_data) {
  if (w > z_bin_counts_[0].size() || w < 0) {
    throw std::runtime_error("Invalid ZBin index!");
  }
  // Just cast the std::vector to the Eigen matrix
  std::unique_lock<std::mutex> lock(mutex_);
  // z_bin_data.resize(2 * z_bin_indices_[w][v * stride_ + u].size());
  for (int i = 0; i < z_bin_indices_[w][v * stride_ + u].size(); ++i) {
    z_bin_data(2 * i, 0) = data_[v * stride_ + u][z_bin_indices_[w][v * stride_ + u][i]];
    z_bin_data(2 * i + 1, 0) = data_[v * stride_ + u][z_bin_indices_[w][v * stride_ + u][i] + 1];
  }
}

void DepthCalibrator::GetZBinIndices(int v, int u, int w, Eigen::Ref<DepthCalibrator::MRowidx> z_bin_indices) {
  if (w > z_bin_counts_[0].size() || w < 0) {
    throw std::runtime_error("Invalid ZBin index!");
  }
  // Just cast the std::vector to the Eigen matrix
  std::unique_lock<std::mutex> lock(mutex_);
  z_bin_indices = Eigen::Map<DepthCalibrator::MRowidx>(z_bin_indices_[w][v * stride_ + u].data(),
                                                       z_bin_indices_[w][v * stride_ + u].size(), 1);
}

void DepthCalibrator::FitBin(int bin) {
  // Do nothing for now.
  std::cout << "Not implemented yet!\n";
}

void DepthCalibrator::SaveData(const std::string &file_name) {
  // Just save the data
  std::unique_lock<std::mutex> lock(mutex_);
  std::fstream savefile;
  savefile = std::fstream(file_name, std::ios::out | std::ios::binary);
  // First save the cell width, cell height, and stride
  savefile.write(reinterpret_cast<char *>(&cell_width_), sizeof(cell_width_));
  savefile.write(reinterpret_cast<char *>(&cell_height_), sizeof(cell_height_));
  savefile.write(reinterpret_cast<char *>(&z_min_), sizeof(z_min_));
  savefile.write(reinterpret_cast<char *>(&z_max_), sizeof(z_max_));
  savefile.write(reinterpret_cast<char *>(&z_bin_stride_), sizeof(z_bin_stride_));
  savefile.write(reinterpret_cast<char *>(&z_bin_count_max_), sizeof(z_bin_count_max_));
  savefile.write(reinterpret_cast<char *>(&stride_), sizeof(stride_));
  size_t dim = data_.size();
  savefile.write(reinterpret_cast<char *>(&dim), sizeof(size_t));
  // And now write out all the data!
  for (size_t i = 0; i < dim; ++i) {
    // Get size of this bin's data
    size_t num_vals = data_[i].size();
    // Write it
    savefile.write(reinterpret_cast<char *>(&num_vals), sizeof(size_t));
    // And fill!
    savefile.write(reinterpret_cast<char *>(data_[i].data()), num_vals * sizeof(float));
  }
  // Also save the z_bin data
  size_t num_z_bins = static_cast<int>(ceilf((z_max_ - z_min_) / z_bin_stride_));
  savefile.write(reinterpret_cast<char *>(&num_z_bins), sizeof(num_z_bins));
  for (size_t i = 0; i < num_z_bins; ++i) {
    savefile.write(reinterpret_cast<char *>(z_bin_counts_[i].data()), dim * sizeof(float));
  }
  savefile.close();
}

void DepthCalibrator::ReadData(const std::string &file_name) {
  // Just read the data
  std::unique_lock<std::mutex> lock(mutex_);
  std::fstream readfile;
  data_.clear();
  readfile = std::fstream(file_name, std::ios::in | std::ios::binary);
  // First save the cell width, cell height, and stride
  readfile.read(reinterpret_cast<char *>(&cell_width_), sizeof(cell_width_));
  readfile.read(reinterpret_cast<char *>(&cell_height_), sizeof(cell_height_));
  readfile.read(reinterpret_cast<char *>(&z_min_), sizeof(z_min_));
  readfile.read(reinterpret_cast<char *>(&z_max_), sizeof(z_max_));
  readfile.read(reinterpret_cast<char *>(&z_bin_stride_), sizeof(z_bin_stride_));
  readfile.read(reinterpret_cast<char *>(&z_bin_count_max_), sizeof(z_bin_count_max_));
  readfile.read(reinterpret_cast<char *>(&stride_), sizeof(stride_));
  size_t dim = 0;
  readfile.read(reinterpret_cast<char *>(&dim), sizeof(size_t));
  std::cout << "Read::\n width::" << cell_width_ << "\n height::" << cell_height_ << "\n z_min::" << z_min_ << "\n z_max::" << z_max_ << "\n z_bin_stride::" << z_bin_stride_ << "\n z_bin_count_max::" << z_bin_count_max_ << "\n stride::" << stride_ << "\n dim::" << dim << "\n";
  data_.resize(dim);
  // And now read out all the data!
  for (size_t i = 0; i < dim; ++i) {
    // Get size of this bin's data
    size_t num_vals = 0;
    readfile.read(reinterpret_cast<char *>(&num_vals), sizeof(size_t));
    data_[i].resize(num_vals);
    // And fill!
    readfile.read(reinterpret_cast<char *>(data_[i].data()), num_vals * sizeof(float));
  }
  // Also fill the z bin counts
  size_t num_z_bins = 0;
  readfile.read(reinterpret_cast<char *>(&num_z_bins), sizeof(size_t));
  z_bin_counts_.resize(num_z_bins);
  z_bin_indices_.resize(num_z_bins);
  for (size_t i = 0; i < num_z_bins; ++i) {
    z_bin_counts_[i].resize(dim);
    z_bin_indices_[i].resize(dim, std::vector<size_t>());
    readfile.read(reinterpret_cast<char *>(z_bin_counts_[i].data()), dim * sizeof(float));
  }
  readfile.close();
  // Populate the z_bin_indices
  for (size_t i = 0; i < dim; ++i) {
    for (size_t idx = 0; idx < data_[i].size(); idx += 2) {
      int z_bin = static_cast<int>((data_[i][idx + 1] - z_min_) / z_bin_stride_);
      z_bin_indices_[z_bin][i].push_back(idx);
    }
  }
}