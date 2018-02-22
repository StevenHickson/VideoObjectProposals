cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 11)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
project(Video_Object_Proposals)

SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

IF(WIN32 OR win64)
   SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib .dll)
ELSE()
   SET(CMAKE_FIND_LIBRARY_SUFFIXES .a)
ENDIF()

find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)

find_path(gflags_dir "gflags" "/usr/include" "/usr/local/include")
find_path(4D_Segmentation_Include_Dir "4D_Segmentation" "~/")
find_library(gflags_lib "libgflags.so" "/usr/lib/x86_64-linux-gnu/" "/usr/lib" "/usr/local/lib")
find_library(4D_Segmentation_lib "lib4D_Segmentation.a" "${4D_Segmentation_Include_Dir}/bin")
 

include_directories(${PCL_INCLUDE_DIRS})
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})
include_directories(${Video_Object_Proposals_SOURCE_DIR}/include)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

include_directories(${Video_Object_Proposals_INCLUDE_DIRS})
include_directories(${gflags_dir})
include_directories(${4D_Segmentation_Include_Dir})

add_library (conversions conversions.cpp conversions.h)
target_link_libraries (conversions ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_library (point_cloud_ops point_cloud_ops.cpp point_cloud_ops.h)
target_link_libraries (point_cloud_ops conversions ${PCL_LIBRARIES} ${OpenCV_LIBS} ${gflags_lib} ${4D_Segmentation_lib})

add_library (image_extractor image_extractor.cpp image_extractor.h city_scape_info.h)
target_link_libraries (image_extractor conversions ${PCL_LIBRARIES} ${OpenCV_LIBS} ${gflags_lib} ${4D_Segmentation_lib})

add_executable (generate_instances generate_instances.cpp point_cloud_ops.h city_scape_info.h)
target_link_libraries (generate_instances conversions point_cloud_ops ${PCL_LIBRARIES} ${OpenCV_LIBS} ${gflags_lib} ${4D_Segmentation_lib})
