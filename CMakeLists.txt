cmake_minimum_required(VERSION 3.11)
# Make sure to include CUDA, so that GVDB header files are processed correctly,
# and C, so that GLEW compiles correctly:
project(mrfmap_ros LANGUAGES CUDA CXX C)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED TRUE)

# Find catkin macros and libraries if COMPONENTS list like find_package(catkin
# REQUIRED COMPONENTS xyz) is used, also find other catkin packages
find_package(
  catkin REQUIRED
  COMPONENTS cv_bridge
             geometry_msgs
             image_transport
             rosbag
             roscpp
             rospy
             sensor_msgs
             tf2_ros)
# System dependencies are found with CMake's conventions Specify Eigen path from
# ENV
set(EIGEN3_INCLUDE_DIR
    "$ENV{EIGEN3_INCLUDE_DIR}"
    CACHE FILEPATH "Path to Eigen 3.3.7")
find_package(Eigen3 3.3.7 REQUIRED)
find_package(OpenCV REQUIRED)

set(PythonLibsNew_FIND_VERSION 2.7)
find_package(Python2 COMPONENTS Development REQUIRED)
# Uncomment this if the package has a setup.py. This macro ensures modules and
# global scripts declared therein get installed See
# http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
#
catkin_python_setup()
# ##############################################################################
# catkin specific configuration ##
# ##############################################################################
# The catkin_package macro generates cmake config files for your package Declare
# things to be passed to dependent projects INCLUDE_DIRS: uncomment this if your
# package contains header files LIBRARIES: libraries you create in this project
# that dependent projects also need CATKIN_DEPENDS: catkin_packages dependent
# projects also need DEPENDS: system dependencies of this project that dependent
# projects also need
catkin_package(
  INCLUDE_DIRS
  include
  # LIBRARIES mrfmap_ros
  CATKIN_DEPENDS
  cv_bridge
  geometry_msgs
  image_transport
  rosbag
  roscpp
  rospy
  sensor_msgs
  tf2_ros
  DEPENDS
  OpenCV)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ffast-math  -fopenmp -O3 -fPIC -march=native")
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math  -fopenmp -O3 -fPIC -march=native"
)# -march=native required because gtsam uses it for compilation otherwise Eigen
 # gets compiled with two different options

set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG}  -fopenmp -fPIC -march=native -g3")
# ### Fetch the mrfmap library ###
set(mrfmap_PATH "/home/icoderaven/packages/mrfmap/install/share/cmake/")
find_package(mrfmap REQUIRED PATHS ${mrfmap_PATH})

# Combine common system include directories that we end up using
set(COMMON_INCLUDE_DIRS ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
                        ${YAML_CPP_INCLUDEDIR} ${OPENCV_INCLUDE_DIRS})
# ##############################################################################
# Build ##
# ##############################################################################

add_executable(GVDBBatchMapCreator src/batch_map_creator.cpp
                                   src/GVDBBatchMapCreator.cpp)
target_link_libraries(
  GVDBBatchMapCreator PRIVATE mrfmap::mrfmap mrfmap::pango_viewer
                              mrfmap::codetimer ${catkin_LIBRARIES})
target_include_directories(GVDBBatchMapCreator PRIVATE include
                                                       ${catkin_INCLUDE_DIRS})

add_executable(ros_mrfmap_node src/ros_mrfmap_node.cpp)
target_link_libraries(
  ros_mrfmap_node PRIVATE mrfmap::mrfmap mrfmap::pango_viewer
                          ${catkin_LIBRARIES})
target_include_directories(ros_mrfmap_node PRIVATE include
                                                   ${catkin_INCLUDE_DIRS})

# Add pybind11 wrapper and link it to GVDBPyModules to access all of them

# get_target_property(SHARED_LIB_PATH mrfmap::mrfmap IMPORTED_LOCATION_RELEASE)
# get_filename_component(DIR ${SHARED_LIB_PATH} DIRECTORY) set(pybind11_DIR
# "${DIR}/../share/cmake/pybind11") message("pybind dir is ${pybind11_DIR}")
# find_package(pybind11 REQUIRED)

add_library(MRFMapRosPyModules MODULE src/MRFMapRosPyModules.cpp
                                      src/GVDBBatchMapCreator.cpp)
target_link_libraries(MRFMapRosPyModules mrfmap::mrfmap mrfmap::pango_viewer
                      ${catkin_LIBRARIES} mrfmap::GVDBPyModules mrfmap::module)
set_target_properties(
  MRFMapRosPyModules PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE
                                PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX ".so")
target_include_directories(
  MRFMapRosPyModules PRIVATE include ${catkin_INCLUDE_DIRS}
                             ${Python2_INCLUDE_DIRS})

# ##############################################################################
# Install ##
# ##############################################################################

install(
  TARGETS MRFMapRosPyModules
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})

# Also copy over all the libraries from the targets into lib
# Ugly, but catkin forces my hand...
get_target_property(SHARED_LIB_PATH mrfmap::mrfmap IMPORTED_LOCATION_RELEASE)
get_filename_component(DIR ${SHARED_LIB_PATH} DIRECTORY)
install(
  DIRECTORY ${DIR}/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  FILES_MATCHING
  PATTERN "*.so*")
install(
    DIRECTORY ${DIR}/../include/mrfmap
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include
    FILES_MATCHING
    PATTERN "*.h")

get_target_property(GVDBPY_LIB_PATH mrfmap::GVDBPyModules
                    IMPORTED_LOCATION_RELEASE)
install(
  FILES ${GVDBPY_LIB_PATH}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})

install(
  TARGETS GVDBBatchMapCreator ros_mrfmap_node
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY urdf DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

# Also install the gazebo model to the local gazebo folder install(DIRECTORY
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory $ENV{HOME}/.gazebo/models/${PROJECT_NAME})
install(DIRECTORY models/ DESTINATION $ENV{HOME}/.gazebo/models/${PROJECT_NAME})

install(DIRECTORY worlds DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
