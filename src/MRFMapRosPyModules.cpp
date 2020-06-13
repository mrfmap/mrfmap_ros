#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>

#include <mrfmap_ros/GVDBBatchMapCreator.h>
#include <mrfmap_ros/DepthCalibrator.h>

PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<uint>>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<float>>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<uint>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<float>>);
PYBIND11_MAKE_OPAQUE(std::vector<uint>);
namespace py = pybind11;

gvdb_params_t gvdb_params_;

void set_from_python(py::object obj) {
  gvdb_params_t *cls = obj.cast<gvdb_params_t *>();
  gvdb_params_ = *cls;
  LOG(std::cout << gvdb_params_ << "\n";);
}

PYBIND11_MODULE(MRFMapRosPyModules, m) {
  m.doc() = "Python Wrapper for Occupancy inference using GVDB";  // optional module
                                                                  // docstring
  pybind11::bind_vector<std::vector<float>>(m, std::string("VecFloat"));
  pybind11::bind_vector<std::vector<int>>(m, std::string("VecInt"));
  pybind11::bind_vector<std::vector<uint>>(m, std::string("VecUInt"));
  pybind11::bind_vector<std::vector<std::vector<uint>>>(m, std::string("VecVecUInt"));
  pybind11::bind_vector<std::vector<std::vector<float>>>(m, std::string("VecVecFloat"));
  pybind11::bind_vector<std::vector<std::vector<std::vector<uint>>>>(m, std::string("VecVecVecUInt"));
  pybind11::bind_vector<std::vector<std::vector<std::vector<float>>>>(m, std::string("VecVecVecFloat"));

  pybind11::class_<DepthCalibrator>(m, "DepthCalibrator")
      .def(pybind11::init<int, int, float, float, float, int>())
      .def("add_image", &DepthCalibrator::AddImage)
      .def("get_bin_size", &DepthCalibrator::GetBinSize)
      .def("get_bin_data", &DepthCalibrator::GetBinData)
      .def("get_zbin_size", &DepthCalibrator::GetZBinSize)
      .def("get_zbin_data", &DepthCalibrator::GetZBinData)
      .def("get_zbin_counts", &DepthCalibrator::GetZBinCounts)
      .def("get_zbin_indices", &DepthCalibrator::GetZBinIndices)
      .def("fit_bin", &DepthCalibrator::FitBin)
      .def("read_data", &DepthCalibrator::ReadData)
      .def("save_data", &DepthCalibrator::SaveData);

  pybind11::class_<GVDBBatchMapCreator>(m, "GVDBBatchMapCreator")
      .def(pybind11::init<std::string>())
      .def("load_bag", &GVDBBatchMapCreator::load_bag)
      .def("process_saved_tuples", &GVDBBatchMapCreator::process_saved_tuples)
      .def("add_keyframe", &GVDBBatchMapCreator::add_keyframe)
      .def("get_inference_ptr", &GVDBBatchMapCreator::get_inference_ptr)
      .def("get_octomap_ptr", &GVDBBatchMapCreator::get_octomap_ptr)
      .def("get_stats", &GVDBBatchMapCreator::get_stats);
}