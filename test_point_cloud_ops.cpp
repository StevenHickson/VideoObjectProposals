// Tests the point cloud ops to make sure they work
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <regex>
#include <gflags/gflags.h>
#include "point_cloud_ops.h"
#include "conversions.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/ModelCoefficients.h>

DEFINE_string(filename, "", "The name of the file to parse.");

DEFINE_double(disp_mult, 0.209313, "The number to mulitply the disparity by.");

DEFINE_int32(side_cut_off, 20, "The number of pixels on each side to exclude.");

DEFINE_double(depth_cut_off, 80000,
              "The pixels to exclude greater than a set depth.");

DEFINE_double(
    height_cut_off, 7500,
    "The pixels to exclude over a certain height from the ground plane.");

DEFINE_int32(inlier_dist, 700,
             "The inlier distance for PCL ground plane estimation.");

DEFINE_int32(plane_angle, 0,
             "The plane angle parameter for PCL ground plane estimation.");

DEFINE_int32(focal_length, 2260,
             "The focal length parameter for the depth image.");

DEFINE_int32(image_width, 512, "The width of the depth image.");

DEFINE_int32(image_height, 256, "The height of the depth image.");

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  string disp_name, inst_name, label_name;
  cv::Mat image, disp, depth;
  pcl::PointCloud<pcl::PointXYZRGBA> cloud, output_cloud;

  image = imread(FLAGS_filename, CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
      cout << "Error, couldn't load image" << endl;
      return 1;
  }

  disp_name = regex_replace(FLAGS_filename, regex("png"), "flt");
  if (!ReadFloatImage(disp_name, FLAGS_image_width, FLAGS_image_height, &disp)) {
      printf("Error, couldn't load flt image\n");
      return 1;
  }
  point_cloud_ops::PCOps pcops;
  pcops.SetSegmentationOptions();

  FileStorage output("test_out.yml", cv::FileStorage::WRITE);
  // Let's save the float image in a different format
  output << "disp" << disp;
  
  // Let's generate and save the depth image
  pcops.DisparityToDepth(disp, disp.cols, disp.rows, FLAGS_focal_length, FLAGS_disp_mult, &depth);
  output << "depth" << depth;
  
  // Let's save the point cloud
  pcops.CreatePointCloud(image, depth, &cloud, FLAGS_focal_length);
  pcl::io::savePCDFileASCII("tmp_cloud.pcd", cloud);
  pcl::io::savePLYFileASCII("tmp_cloud.ply", cloud);
  
  // Let's compute the ground plane
  pcl::PointIndices inliers;
  pcops.ComputeGroundPlane(cloud, FLAGS_inlier_dist, FLAGS_plane_angle, 0, 0, &inliers);
  //std::vector<int>::const_iterator pI = inliers.indices.begin();
  //while(pI != inliers.indices.end()) {
  //    cout << *pI++ << endl;
  //}
  Mat inlier_mat(inliers.indices);
  output << "inliers" << inlier_mat;

  // Let's trim the cloud
  pcops.TrimDepth(cloud, inliers, FLAGS_depth_cut_off, FLAGS_side_cut_off, FLAGS_height_cut_off, &output_cloud);
  pcl::io::savePCDFileASCII("output_cloud.pcd", output_cloud);
  pcl::io::savePLYFileASCII("output_cloud.ply", output_cloud);
  return 0;
}
