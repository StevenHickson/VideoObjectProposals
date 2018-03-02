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

include_directories(${PCL_INCLUDE_DIRS})
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})
include_directories(${Video_Object_Proposals_SOURCE_DIR}/include)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

include_directories(${Video_Object_Proposals_INCLUDE_DIRS})
include_directories(${gflags_dir})
include_directories(${4D_Segmentation_Include_Dir})

# CLIF-BEGIN
#[[
# We need the Python libraries for generating PyCLIF wrappers.
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include(FindPkgConfig)
# Lookup include and library directories using pkg-config.
pkg_check_modules(GOOGLE_PROTOBUF REQUIRED protobuf)

find_program(PROTOC "protoc")
if(PROTOC-NOTFOUND)
  message(FATAL_ERROR "The protobuf compiler 'protoc' not found.")
endif(PROTOC-NOTFOUND)

# Path to the "pyclif" binary should either be passed as -DPYCLIF= argument to
# cmake, or should be in the path.
if(NOT PYCLIF)
  find_program(PYCLIF pyclif)
  if(NOT PYCLIF)
    message(FATAL_ERROR "The 'pyclif' program was not found. Specify with -DPYCLIF.")
  endif(NOT PYCLIF)
endif(NOT PYCLIF)
string(REGEX REPLACE "/bin/pyclif$" "" CLIF_INSTALL_DIR ${PYCLIF})

if(NOT PYCLIF_PROTO)
  find_program(PYCLIF_PROTO pyclif_proto)
  if(NOT PYCLIF_PROTO)
    message(FATAL_ERRPR "The 'pyclif_proto' program was not found. Specify with -DPYCLIF_PROTO.")
  endif(NOT PYCLIF_PROTO)
endif(NOT PYCLIF_PROTO)

set(CLIF_EXAMPLES_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CLIF_EXAMPLES_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})


function(add_pyclif_library name pyclif_file)
  cmake_parse_arguments(PYCLIF_LIBRARY "" "" "CC_DEPS;CLIF_DEPS;PROTO_DEPS;CXX_FLAGS" ${ARGN})

  string(REPLACE ".clif" "" pyclif_file_basename ${pyclif_file})
  set(gen_cc "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}.cc")
  set(gen_h "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}.h")
  set(gen_init "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}_init.cc")

  string(REPLACE "-" "_" module_name ${pyclif_file_basename})

  if (GOOGLE_PROTOBUF_INCLUDE_DIRS)
    set(GOOGLE_PROTOBUF_CXX_FLAGS "-I${GOOGLE_PROTOBUF_INCLUDE_DIRS}")
  endif(GOOGLE_PROTOBUF_INCLUDE_DIRS)

  #string(REPLACE " " " -I" ALL_PCL_INCLUDE_DIRS ${PCL_INCLUDE_DIRS})

  add_custom_target(
    ${name}
    BYPRODUCTS ${gen_cc} ${gen_h} ${gen_init}
    COMMAND
      ${PYCLIF}
      # If this is not run in a virtualenv after INSTALL.sh, then the following
      # two lines should be uncommented to point to the correct types.h and
      # the matcher binary.
      -p${CLIF_INSTALL_DIR}/python/types.h
      --matcher_bin=${CLIF_INSTALL_DIR}/clang/bin/clif-matcher
      -c${gen_cc} -g${gen_h} -i${gen_init}
      # Specify the path to the generated files.
      -I${CLIF_EXAMPLES_BINARY_DIR}
      -I${PCL_INCLUDE_DIRS} 
      --modname=${module_name}
      "-f-I${PYTHON_INCLUDE_DIRS} -I${CLIF_INSTALL_DIR}/.. -I${CLIF_EXAMPLES_SOURCE_DIR} -I${CLIF_EXAMPLES_BINARY_DIR} ${GOOGLE_PROTOBUF_CXX_FLAGS} -std=c++11 ${PYCLIF_LIBRARY_CXX_FLAGS}"
      ${CMAKE_CURRENT_SOURCE_DIR}/${pyclif_file}
    VERBATIM
    DEPENDS ${PYCLIF_LIBRARY_CC_DEPS} ${PYCLIF_LIBRARY_CLIF_DEPS} ${PYCLIF_LIBRARY_PROTO_DEPS}
  )
endfunction(add_pyclif_library)
]]
# CLIF-END

find_path(gflags_dir "gflags" "/usr/include" "/usr/local/include")
find_path(4D_Segmentation_Include_Dir "4D_Segmentation" "~/")
find_library(gflags_lib "libgflags.so" "/usr/lib/x86_64-linux-gnu/" "/usr/lib" "/usr/local/lib")
find_library(4D_Segmentation_lib "lib4D_Segmentation.a" "${4D_Segmentation_Include_Dir}/bin")
 
add_library (conversions conversions.cpp conversions.h)
target_link_libraries (conversions ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_library (point_cloud_ops point_cloud_ops.cpp point_cloud_ops.h)
target_link_libraries (point_cloud_ops conversions ${PCL_LIBRARIES} ${OpenCV_LIBS} ${gflags_lib} ${4D_Segmentation_lib})

#add_pyclif_library(point_cloud_ops_clif point_cloud_ops.clif [CC_DEPS point_cloud_ops])

add_library (image_extractor image_extractor.cpp image_extractor.h city_scape_info.h)
target_link_libraries (image_extractor conversions ${PCL_LIBRARIES} ${OpenCV_LIBS} ${gflags_lib} ${4D_Segmentation_lib})

add_executable (generate_instances generate_instances.cpp point_cloud_ops.h city_scape_info.h)
target_link_libraries (generate_instances conversions point_cloud_ops ${PCL_LIBRARIES} ${OpenCV_LIBS} ${gflags_lib} ${4D_Segmentation_lib})

add_executable (test_point_cloud_ops test_point_cloud_ops.cpp point_cloud_ops.h city_scape_info.h)
target_link_libraries (test_point_cloud_ops conversions point_cloud_ops ${PCL_LIBRARIES} ${OpenCV_LIBS} ${gflags_lib} ${4D_Segmentation_lib})

add_executable (compute_metrics compute_metrics.cpp conversions.h city_scape_info.h)
target_link_libraries (compute_metrics ${PCL_LIBRARIES} ${OpenCV_LIBS} ${gflags_lib} ${4D_Segmentation_lib})
