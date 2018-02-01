#include "point_cloud_ops.h"

//#include <type_traits>
#include <gflags/gflags.h>
#include <pcl/ModelCoefficients.h>
#include <iostream>
#include <fstream>

DEFINE_int32(center_of_projection_width, 1024,
             "The center of projection (width) for the point cloud");
DEFINE_int32(center_of_projection_height, 512,
             "The center of projection (height) for the point cloud");

DEFINE_double(sigma_depth, 2.5f,
              "The sigma for gaussian smoothing of the depth");
DEFINE_double(sigma_color, 0.8f,
              "The sigma for gaussian smoothing of the color");
DEFINE_double(sigma_normals, 0.8f,
              "The sigma for gaussian smoothing of the normals");
DEFINE_double(c_depth, 20000,
              "The c constant for FH segmentation of the depth");
DEFINE_double(c_color, 100000,
              "The c constant for FH segmentation of the color");
DEFINE_double(c_normals, 20000,
              "The c constant for FH segmentation of the normals");
DEFINE_int32(depth_min_size, 15000,
             "The c constant for FH segmentation of the depth");
DEFINE_int32(color_min_size, 15000,
             "The c constant for FH segmentation of the color");
DEFINE_int32(normals_min_size, 15000,
             "The c constant for FH segmentation of the normals");
DEFINE_double(max_depth, 80000, "The maximum expected depth");
DEFINE_bool(use_normals, false,
            "Whether to use normal estimation instead of depth");
DEFINE_bool(use_time, false,
            "Whether to use temporal information (number_of_frames)");
DEFINE_bool(use_fast_method, false,
            "Whether to use the faster(less accurate) method");
DEFINE_int32(number_of_frames, 1, "The number of frames to segment at once");

using cv::Mat;
using cv::Mat_;
using cv::Vec3b;
using pcl::ModelCoefficients;
using pcl::Normal;
using pcl::PointCloud;
using pcl::PointIndices;
using pcl::PointXYZI;
using pcl::PointXYZRGBA;
using pcl::SACSegmentation;

namespace {

int Clamp(int val, int min, int max) {
  if (val > max) return max;
  if (val < min) return min;
  return val;
}

void VectorToMatColor(const std::vector<int>& input, const int width,
                      const int height, const int nChannels, Mat* output) {
  if (output == NULL) {
    std::cout << "Output Mat cannot be NULL in VectorToMatColor" << std::endl;
    return;
  }
  *output = Mat(height, width, CV_8UC3);
  typename Mat_<Vec3b>::iterator pO = output->begin<Vec3b>();
  typename std::vector<int>::const_iterator p = input.begin();
  while (p != input.end()) {
    *pO++ = Vec3b(static_cast<uchar>(*p++), static_cast<uchar>(*p++),
                  static_cast<uchar>(*p++));
  }
}

void VectorToMatFloat(const std::vector<float>& input, const int width,
                      const int height, const int nChannels, Mat* output) {
  if (output == NULL) {
    std::cout << "Output Mat cannot be NULL in VectorToMatFloat" << std::endl;
    return;
  }
  *output = Mat(height, width, CV_32F);
  typename Mat_<float>::iterator pO = output->begin<float>();
  typename std::vector<float>::const_iterator p = input.begin();
  while (p != input.end()) {
    *pO++ = static_cast<float>(*p++);
  }
}

void MatToVectorFloat(const Mat& input, const int n_channels,
                      std::vector<float>* output) {
  if (output == NULL) {
    std::cout << "Output Mat cannot be NULL in MatToVectorFloat" << std::endl;
    return;
  }
  output->resize(input.rows * input.cols * n_channels);
  typename Mat_<float>::const_iterator p = input.begin<float>();
  typename std::vector<float>::iterator pO = output->begin();
  while (p != input.end<float>()) {
    *pO++ = static_cast<float>(*p++);
  }
}

void MatToVectorColor(const Mat& input, std::vector<int>* output) {
  if (output == NULL) {
    std::cout << "Output Mat cannot be NULL in MatToVectorColor" << std::endl;
    return;
  }
  output->resize(input.rows * input.cols * 3);
  typename Mat_<Vec3b>::const_iterator p = input.begin<Vec3b>();
  typename std::vector<int>::iterator pO = output->begin();
  while (p != input.end<Vec3b>()) {
    *pO++ = static_cast<int>((*p)[0]);
    *pO++ = static_cast<int>((*p)[1]);
    *pO++ = static_cast<int>((*p)[2]);
    p++;
  }
}

void MatToVectorInt(const Mat& input, std::vector<int>* output) {
  if (output == NULL) {
    std::cout << "Output Mat cannot be NULL in MatToVectorInt" << std::endl;
    return;
  }
  output->resize(input.rows * input.cols);
  typename Mat_<int>::const_iterator p = input.begin<int>();
  typename std::vector<int>::iterator pO = output->begin();
  while (p != input.end<int>()) {
    *pO++ = static_cast<int>(*p++);
  }
}

void PointCloudToVectorFloat(const PointCloud<PointXYZRGBA>& pc,
                             std::vector<float>* output) {
  if (output == NULL) {
    std::cout << "Output Mat cannot be NULL in PointCloudToVectorFloat" << std::endl;
    return;
  }
  output->resize(pc.width * pc.height * 3);
  PointCloud<PointXYZRGBA>::const_iterator pCloud = pc.begin();
  typename std::vector<float>::iterator pO = output->begin();
  while (pCloud != pc.end()) {
    *pO++ = pCloud->x;
    *pO++ = pCloud->y;
    *pO++ = pCloud->z;
    pCloud++;
  }
}

void PointCloudToVectorFloat(const PointCloud<Normal>& pc,
                             std::vector<float>* output) {
  if (output == NULL) {
    std::cout << "Output Mat cannot be NULL in PointCloudToVectorFloat" << std::endl;
    return;
  }
  output->resize(pc.width * pc.height * 3);
  PointCloud<Normal>::const_iterator pCloud = pc.begin();
  typename std::vector<float>::iterator pO = output->begin();
  while (pCloud != pc.end()) {
    *pO++ = pCloud->normal_x;
    *pO++ = pCloud->normal_y;
    *pO++ = pCloud->normal_z;
    pCloud++;
  }
}

void PointCloudToVectorInt(const PointCloud<PointXYZI>& pc,
                           std::vector<int>* output) {
  if (output == NULL) {
    std::cout << "Output Mat cannot be NULL in PointCloudToVectorInt" << std::endl;
    return;
  }
  output->resize(pc.width * pc.height * 3);
  PointCloud<PointXYZI>::const_iterator pCloud = pc.begin();
  typename std::vector<int>::iterator pO = output->begin();
  while (pCloud != pc.end()) {
    *pO++ = pCloud->x;
    *pO++ = pCloud->y;
    *pO++ = pCloud->z;
    *pO++ = static_cast<int>(pCloud->intensity);
    pCloud++;
  }
}

void PointCloudToVectorColor(const PointCloud<PointXYZRGBA>& pc,
                             std::vector<int>* output) {
  if (output == NULL) {
    std::cout << "Output Mat cannot be NULL in PointCloudToVectorColor" << std::endl;
    return;
  }
  output->resize(pc.width * pc.height * 3);
  PointCloud<PointXYZRGBA>::const_iterator pCloud = pc.begin();
  typename std::vector<int>::iterator pO = output->begin();
  while (pCloud != pc.end()) {
    *pO++ = static_cast<int>(pCloud->r);
    *pO++ = static_cast<int>(pCloud->g);
    *pO++ = static_cast<int>(pCloud->b);
    pCloud++;
  }
}

}  // namespace

namespace point_cloud_ops {

void PCOps::ComputeOpticalFlow(const Mat& past, const Mat& current,
                               Mat* output) {
  Mat in1, in2;
  cv::cvtColor(past, in1, CV_BGR2GRAY);
  cv::cvtColor(current, in2, CV_BGR2GRAY);
  cv::calcOpticalFlowFarneback(in1, in2, *output, 0.5f, 2, 5, 2, 7, 1.5, 0);
}

void PCOps::ComputeOpticalFlow3D(const Mat& past, const Mat& current,
                                 const PointCloud<PointXYZRGBA>& past_cloud,
                                 const PointCloud<PointXYZRGBA>& current_cloud,
                                 const float focal_length,
                                 PointCloud<Normal>* output) {
  Mat flow2d;
  ComputeOpticalFlow(past, current, &flow2d);
  output->height = flow2d.rows;
  output->width = flow2d.cols;
  output->is_dense = false;
  output->resize(output->height * output->width);
  output->sensor_origin_.setZero();
  PointCloud<Normal>::iterator pOut = output->begin();
  PointCloud<PointXYZRGBA>::const_iterator pCloud = current_cloud.begin();
  Mat_<cv::Vec2f>::iterator pIn = flow2d.begin<cv::Vec2f>();
  int safeWidth = current_cloud.width - 1,
      safeHeight = current_cloud.height - 1;
  for (int j = 0; j < current_cloud.height; j++) {
    for (int i = 0; i < current_cloud.width; i++) {
      pOut->normal_x = (*pIn)[0] * pCloud->z * focal_length;
      pOut->normal_y = (*pIn)[1] * pCloud->z * focal_length;
      pOut->normal_z =
          (pCloud->z -
           past_cloud(Clamp(static_cast<int>(i - (*pIn)[0]), 0, safeWidth),
                      Clamp(static_cast<int>(j - (*pIn)[1]), 0, safeHeight))
               .z);
      ++pIn;
      ++pOut;
      ++pCloud;
    }
  }
}

void PCOps::LoadNPY(const std::string& filename, int* width, int* height,
                    std::vector<float>* output) {
  std::string bytes;
  std::ifstream file(filename.c_str(), std::ifstream::binary);
  *width = 512;
  *height = 256;
  int size = *height * *width;
  output->resize(size);
  if (output->size() < size) {
    std::cout << "Could not allocate memory for NPY" << std::endl;
    return;
  }

  memcpy(&*output->begin(), file.rdbuf(), size * sizeof(float));
}

void PCOps::SetSegmentationOptions() {
  if (options_ != NULL) {
    delete options_;
  }
  options_ = new SegmentationOptions();
  if (options_ == NULL) {
    std::cout << "Couldn't allocate SegmentationOptions." << std::endl;
    return;
  }
  options_->sigma_depth = FLAGS_sigma_depth;
  options_->sigma_color = FLAGS_sigma_color;
  options_->sigma_normals = FLAGS_sigma_normals;
  options_->c_depth = FLAGS_c_depth;
  options_->c_color = FLAGS_c_color;
  options_->c_normals = FLAGS_c_normals;
  options_->depth_min_size = FLAGS_depth_min_size;
  options_->color_min_size = FLAGS_color_min_size;
  options_->normals_min_size = FLAGS_normals_min_size;
  options_->max_depth = FLAGS_max_depth;
  options_->use_normals = FLAGS_use_normals;
  options_->use_time = FLAGS_use_time;
  options_->use_fast_method = FLAGS_use_fast_method;
  options_->number_of_frames = FLAGS_number_of_frames;

  if (segments_ != NULL) {
    delete segments_;
  }
  segments_ = new RGBDTSegmentation(*options_);
}

void PCOps::GetSegments(const PointCloud<PointXYZRGBA>& cloud,
                        PointCloud<PointXYZI>* labels,
                        PointCloud<PointXYZRGBA>* labels_color) {
  // The library uses non-const boost shared_ptrs in their function calls.
  // We use makeShared from PCL and copy the point clouds manually.
  // This is inefficient but makes the code easier to read and maintain.
  if (labels == NULL || labels_color == NULL) {
    std::cout << "Output can't be null in GetSegments." << std::endl;
    return;
  }

  if (segments_ == NULL) {
    std::cout
        << "SetSegmentationOptions must be called once before GetSegments." << std::endl;
    return;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr labels_internal(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr labels_color_internal(
      new pcl::PointCloud<pcl::PointXYZRGBA>);
  segments_->AddSlice(cloud.makeShared(), labels_internal,
                      labels_color_internal);
  copyPointCloud(*labels_internal, *labels);
  copyPointCloud(*labels_color_internal, *labels_color);
}

void PCOps::DisparityToDepth(const Mat& input_disparity, const int new_width,
                             const int new_height, const float focal_point,
                             const float disp_mult, Mat* output_depth) {
  if (output_depth == NULL) {
    std::cout << "Output Mat cannot be NULL in DisparityToDepth" << std::endl;
    return;
  }

  // Resize to color size first
  resize(input_disparity, *output_depth, cv::Size(new_width, new_height));
  // For cityscapes, focal_point is 2260 and disp_nmult is 0.209313
  *output_depth = focal_point * disp_mult / *output_depth;
}

void PCOps::TrimDepth(const PointCloud<PointXYZRGBA>& input_cloud,
                      const PointIndices& ground_plane,
                      const float depth_cut_off, const int side_cut_off,
                      const float height_cut_off,
                      PointCloud<PointXYZRGBA>* output_cloud) {
  if (output_cloud == NULL) {
    std::cout << "Output Cloud cannot be NULL in TrimDepth" << std::endl;
    return;
  }
  copyPointCloud(input_cloud, *output_cloud);
  // Let's cut off all the ground plane first
  std::vector<int>::const_iterator pInd = ground_plane.indices.begin();
  while (pInd != ground_plane.indices.end()) {
    output_cloud->points[*pInd].x = 0;
    output_cloud->points[*pInd].y = 0;
    output_cloud->points[*pInd].z = 0;
    output_cloud->points[*pInd].r = 0;
    output_cloud->points[*pInd].g = 0;
    output_cloud->points[*pInd].b = 0;
    ++pInd;
  }

  // Let's now cut off above the height_cut_off.
  std::vector<int> too_high;
  GetCutOffInliers(input_cloud, height_cut_off, ground_plane.indices, true,
                   &too_high);
  pInd = too_high.begin();
  while (pInd != too_high.end()) {
    output_cloud->points[*pInd].x = 0;
    output_cloud->points[*pInd].y = 0;
    output_cloud->points[*pInd].z = 0;
    output_cloud->points[*pInd].r = 0;
    output_cloud->points[*pInd].g = 0;
    output_cloud->points[*pInd].b = 0;
    ++pInd;
  }

  // Now let's cut off anything past the depth cut off and cut off the sides
  PointCloud<PointXYZRGBA>::iterator pCloud = output_cloud->begin();
  const int left_cut = 0 + side_cut_off;
  const int right_cut = output_cloud->width - side_cut_off - 1;
  for (int j = 0; j < output_cloud->height; j++) {
    for (int i = 0; i < output_cloud->width; i++, pCloud++) {
      if (pCloud->z >= depth_cut_off || i < left_cut || i > right_cut) {
        pCloud->x = pCloud->y = pCloud->z = pCloud->r = pCloud->g = pCloud->b =
            0;
      }
    }
  }
}

void PCOps::CreatePointCloud(const Mat& input_image, const Mat& input_depth,
                             PointCloud<PointXYZRGBA>* cloud, float focal) {
  if (cloud == NULL) {
    std::cout << "cloud cannot be NULL in CreatePointCloud" << std::endl;
    return;
  }
  cloud->header.frame_id = "/cityscapes_frame";
  cloud->height = input_image.rows;
  cloud->width = input_image.cols;
  cloud->is_dense = true;
  cloud->points.resize(cloud->width * cloud->height);

  PointCloud<PointXYZRGBA>::iterator pCloud = cloud->begin();
  Mat_<Vec3b>::const_iterator pImg = input_image.begin<Vec3b>();
  Mat_<float>::const_iterator pDepth = input_depth.begin<float>();
  for (int j = 0; j < input_image.rows; j++) {
    for (int i = 0; i < input_image.cols; i++, pCloud++, pImg++, pDepth++) {
      pCloud->z = *pDepth;
      pCloud->x = static_cast<float>(i - FLAGS_center_of_projection_width) *
                  *pDepth / focal;
      pCloud->y = static_cast<float>(j - FLAGS_center_of_projection_height) *
                  *pDepth / focal;
      pCloud->r = (*pImg)[2];
      pCloud->g = (*pImg)[1];
      pCloud->b = (*pImg)[0];
    }
  }
  cloud->sensor_origin_.setZero();
  cloud->sensor_orientation_.w() = 1.0;
  cloud->sensor_orientation_.x() = 0;
  cloud->sensor_orientation_.y() = 0;
  cloud->sensor_orientation_.z() = 0;
}

void PCOps::GetMatFromCloud(const PointCloud<PointXYZRGBA>& cloud, Mat* image) {
  if (image == NULL) {
    std::cout << "image cannot be NULL in GetMatFromCloud" << std::endl;
    return;
  }
  *image = Mat(cloud.height, cloud.width, CV_32F);
  Mat_<float>::iterator pImg = image->begin<float>();
  PointCloud<PointXYZRGBA>::const_iterator pCloud = cloud.begin();
  while (pCloud != cloud.end()) {
    *pImg++ = pCloud->z;
    pCloud++;
  }
}

void PCOps::GetLabeledMatFromCloud(const PointCloud<PointXYZI>& cloud,
                                   Mat* image) {
  if (image == NULL) {
    std::cout << "image cannot be NULL in GetLabeledMatFromCloud" << std::endl;
    return;
  }
  *image = Mat(cloud.height, cloud.width, CV_32S);
  Mat_<int>::iterator pImg = image->begin<int>();
  PointCloud<PointXYZI>::const_iterator pCloud = cloud.begin();
  while (pCloud != cloud.end()) {
    *pImg++ = pCloud->intensity;
    pCloud++;
  }
}

void PCOps::GetColoredMatFromCloud(const PointCloud<PointXYZRGBA>& cloud,
                                   Mat* image) {
  if (image == NULL) {
    std::cout << "image cannot be NULL in GetColoredMatFromCloud" << std::endl;
    return;
  }
  *image = Mat(cloud.height, cloud.width, CV_8UC3);
  Mat_<Vec3b>::iterator pImg = image->begin<Vec3b>();
  PointCloud<PointXYZRGBA>::const_iterator pCloud = cloud.begin();
  while (pCloud != cloud.end()) {
    *pImg++ = Vec3b(pCloud->b, pCloud->g, pCloud->r);
    pCloud++;
  }
}

void PCOps::ComputeGroundPlane(const PointCloud<PointXYZRGBA>& input_cloud,
                               const float inlier_dist, const int plane_angle,
                               const int method, const int plane_selection,
                               PointIndices* inliers) {
  if (inliers == NULL) {
    std::cout << "inliers cannot be NULL in ComputeGroundPlane" << std::endl;
    return;
  }
  ModelCoefficients::Ptr coefficients(new ModelCoefficients);
  SACSegmentation<PointXYZRGBA> seg;
  seg.setOptimizeCoefficients(true);
  if (method == 0) {
    seg.setModelType(pcl::SACMODEL_PLANE);
  } else {
    seg.setModelType(pcl::SACMODEL_PARALLEL_PLANE);
    if (plane_selection == 0)
      seg.setAxis(Eigen::Vector3f(1, 0, 0));
    else if (plane_selection == 1)
      seg.setAxis(Eigen::Vector3f(0, 1, 0));
    else
      seg.setAxis(Eigen::Vector3f(0, 0, 1));
    seg.setEpsAngle(plane_angle);
  }
  switch (method) {
    case 1:
      seg.setMethodType(pcl::SAC_LMEDS);
      break;
    case 2:
      seg.setMethodType(pcl::SAC_MSAC);
      break;
    case 3:
      seg.setMethodType(pcl::SAC_RRANSAC);
      break;
    case 4:
      seg.setMethodType(pcl::SAC_RMSAC);
      break;
    case 5:
      seg.setMethodType(pcl::SAC_MLESAC);
      break;
    case 6:
      seg.setMethodType(pcl::SAC_PROSAC);
      break;
    default:
      seg.setMethodType(pcl::SAC_RANSAC);
      break;
  }
  seg.setDistanceThreshold(inlier_dist);
  seg.setInputCloud(input_cloud.makeShared());
  seg.segment(*inliers, *coefficients);
}

void PCOps::GetCutOffInliers(const PointCloud<PointXYZRGBA>& input_cloud,
                             const float height_cut_off,
                             const std::vector<int>& ground_plane,
                             const bool above_plane,
                             std::vector<int>* indices) {
  if (indices == NULL) {
    std::cout << "indices cannot be NULL in GetCutOffInliers" << std::endl;
    return;
  }
  // First let's compute the average height of the ground plane.
  std::vector<int>::const_iterator pInd = ground_plane.begin();
  double ground_height = 0;
  while (pInd != ground_plane.end()) {
    ground_height += input_cloud.points[*pInd].y;
    ++pInd;
  }
  ground_height /= ground_plane.size();
  float total_height_cut_off;
  if (above_plane)
    total_height_cut_off = ground_height - height_cut_off;
  else
    total_height_cut_off = height_cut_off;
  PointCloud<PointXYZRGBA>::const_iterator pCloud = input_cloud.begin();
  int iter = 0;
  while (pCloud != input_cloud.end()) {
    if (pCloud->y < total_height_cut_off) indices->push_back(iter);
    ++pCloud;
    ++iter;
  }
}

void PCOps::GetFilteredCloud(const Mat& input_image, const Mat& input_disp,
                             const float focal, const float disp_mult,
                             const int depth_cut_off, const int side_cut_off,
                             const float height_cut_off,
                             const float inlier_dist, const int plane_angle,
                             pcl::PointCloud<pcl::PointXYZRGBA>* output_cloud) {
  if (output_cloud == NULL) {
    std::cout << "output_cloud cannot be NULL in GetFilteredCloud" << std::endl;
    return;
  }

  pcl::PointCloud<pcl::PointXYZRGBA> cloud;
  pcl::PointIndices inliers;
  Mat depth;

  DisparityToDepth(input_disp, input_image.cols, input_image.rows, focal,
                   disp_mult, &depth);
  CreatePointCloud(input_image, depth, &cloud, focal);
  ComputeGroundPlane(cloud, inlier_dist, plane_angle, 0, 0, &inliers);
  TrimDepth(cloud, inliers, depth_cut_off, side_cut_off, height_cut_off,
            output_cloud);
}

void PCOps::GenerateInstanceImage(
    const Mat& input_image, const Mat& input_disp, const float focal,
    const float disp_mult, const int depth_cut_off, const int side_cut_off,
    const float height_cut_off, const float inlier_dist, const int plane_angle,
    const bool return_color, Mat* output_image) {
  if (output_image == NULL) {
    std::cout << "output_image cannot be NULL in GenerateInstanceImage" << std::endl;
    return;
  }

  pcl::PointCloud<pcl::PointXYZI> labels;
  pcl::PointCloud<pcl::PointXYZRGBA> labels_color, filtered;

  GetFilteredCloud(input_image, input_disp, focal, disp_mult, depth_cut_off,
                   side_cut_off, height_cut_off, inlier_dist, plane_angle,
                   &filtered);
  GetSegments(filtered, &labels, &labels_color);

  if (!labels.empty()) {
    if (return_color) {
      GetColoredMatFromCloud(labels_color, output_image);
    } else {
      GetLabeledMatFromCloud(labels, output_image);
    }
  }
}

void PCOps::GenerateZeroBackgroundInstances(
    const Mat& input_image, const Mat& input_disp, const float focal,
    const float disp_mult, const int depth_cut_off, const int side_cut_off,
    const float height_cut_off, const float inlier_dist, const int plane_angle,
    Mat* output_image) {
  // GenerateInstanceImage checks that output_image is not null.
  GenerateInstanceImage(input_image, input_disp, focal, disp_mult,
                        depth_cut_off, side_cut_off, height_cut_off,
                        inlier_dist, plane_angle, false, output_image);

  // Now let's grab the biggest of the instances and label it as 0 for
  // background.
  Mat_<int>::iterator pImg = output_image->begin<int>();
  std::map<int, int> labelCounts;
  while (pImg != output_image->end<int>()) {
    labelCounts[*pImg++]++;
  }

  // First we need to find the biggest label.
  std::map<int, int>::iterator pCounts = labelCounts.begin();
  int maxArg = pCounts->first;
  int max = pCounts->second;
  while (pCounts != labelCounts.end()) {
    if (pCounts->second > max) {
      maxArg = pCounts->first;
      max = pCounts->second;
    }
    ++pCounts;
  }

  // If the label is 0, we are fine already. Otherwise swap it.
  if (maxArg != 0) {
    pImg = output_image->begin<int>();
    while (pImg != output_image->end<int>()) {
      if (*pImg == 0)
        *pImg = maxArg;
      else if (*pImg == maxArg)
        *pImg = 0;
      ++pImg;
    }
  }
}

void PCOps::ComputeOpticalFlowFromVector(const std::vector<int>& past_image,
                                         const std::vector<int>& current_image,
                                         const int width, const int height,
                                         std::vector<float>* flow) {
  Mat past_image_mat, current_image_mat, flow_mat;

  VectorToMatColor(past_image, width, height, 3, &past_image_mat);
  VectorToMatColor(current_image, width, height, 3, &current_image_mat);

  ComputeOpticalFlow(past_image_mat, current_image_mat, &flow_mat);

  MatToVectorFloat(flow_mat, 2, flow);
}

void PCOps::ComputeOpticalFlow3DFromVector(
    const std::vector<int>& past_image, const std::vector<float>& past_depth,
    const std::vector<int>& current_image,
    const std::vector<float>& current_depth, const int width, const int height,
    const float focal, std::vector<float>* flow) {
  PointCloud<PointXYZRGBA> past_cloud, current_cloud;
  PointCloud<Normal> flow_cloud;
  Mat past_image_mat, past_depth_mat, current_image_mat, current_depth_mat;

  VectorToMatColor(past_image, width, height, 3, &past_image_mat);
  VectorToMatFloat(past_depth, width, height, 1, &past_depth_mat);
  VectorToMatColor(current_image, width, height, 3, &current_image_mat);
  VectorToMatFloat(current_depth, width, height, 1, &current_depth_mat);
  CreatePointCloud(past_image_mat, past_depth_mat, &past_cloud, focal);
  CreatePointCloud(current_image_mat, current_depth_mat, &current_cloud, focal);

  ComputeOpticalFlow3D(past_image_mat, current_image_mat, past_cloud,
                       current_cloud, focal, &flow_cloud);

  PointCloudToVectorFloat(flow_cloud, flow);
}

void PCOps::GetGroundPlaneFromVector(
    const std::vector<int>& image, const std::vector<float>& depth,
    const int width, const int height, const int nChannels, const float focal,
    const float inlier_dist, const int plane_angle, const int method,
    const int plane_selection, std::vector<int>* indices) {
  if (indices == NULL) {
    std::cout << "indices cannot be NULL in GetGroundPlaneFromVector" << std::endl;
    return;
  }
  Mat image_mat, depth_mat;
  pcl::PointCloud<pcl::PointXYZRGBA> cloud;
  pcl::PointIndices inliers;

  VectorToMatColor(image, width, height, nChannels, &image_mat);
  VectorToMatFloat(depth, width, height, 1, &depth_mat);
  CreatePointCloud(image_mat, depth_mat, &cloud, focal);
  ComputeGroundPlane(cloud, inlier_dist, plane_angle, method, plane_selection,
                     &inliers);

  *indices = inliers.indices;
}

void PCOps::CreatePointCloudFromVector(const std::vector<int>& image,
                                       const std::vector<float>& depth,
                                       const int width, const int height,
                                       const float focal,
                                       std::vector<float>* output) {
  Mat image_mat, depth_mat;
  pcl::PointCloud<pcl::PointXYZRGBA> cloud;

  VectorToMatColor(image, width, height, 3, &image_mat);
  VectorToMatFloat(depth, width, height, 1, &depth_mat);
  CreatePointCloud(image_mat, depth_mat, &cloud, focal);

  // PointCloudToVectorFloat checks that output is not null.
  PointCloudToVectorFloat(cloud, output);
}

void PCOps::GetSegmentsFromVector(const std::vector<int>& image,
                                  const std::vector<float>& depth,
                                  const int width, const int height,
                                  const float focal, const bool return_color,
                                  std::vector<int>* output) {
  if (output == NULL) {
    std::cout << "output cannot be NULL in GetSegmentsFromVector" << std::endl;
    return;
  }

  Mat image_mat, depth_mat;
  pcl::PointCloud<pcl::PointXYZRGBA> cloud, labels_color;
  pcl::PointCloud<pcl::PointXYZI> labels;

  VectorToMatColor(image, width, height, 3, &image_mat);
  VectorToMatFloat(depth, width, height, 1, &depth_mat);
  CreatePointCloud(image_mat, depth_mat, &cloud, focal);

  GetSegments(cloud, &labels, &labels_color);
  if (!labels.empty()) {
    // PointCloudToVector{Color,Int} checks that output is not null.
    if (return_color) {
      PointCloudToVectorColor(labels_color, output);
    } else {
      PointCloudToVectorInt(labels, output);
    }
  }
}

void PCOps::GetCutOffInliersFromVector(
    const std::vector<int>& image, const std::vector<float>& depth,
    const int width, const int height, const float focal,
    const float height_cut_off, const std::vector<int>& ground_plane,
    const bool above_plane, std::vector<int>* output) {
  Mat image_mat, depth_mat;
  pcl::PointCloud<pcl::PointXYZRGBA> cloud;

  VectorToMatColor(image, width, height, 3, &image_mat);
  VectorToMatFloat(depth, width, height, 1, &depth_mat);
  CreatePointCloud(image_mat, depth_mat, &cloud, focal);

  // GetCutOffInliers checks that output is not null.
  GetCutOffInliers(cloud, height_cut_off, ground_plane, above_plane, output);
}

void PCOps::GetFilteredCloudFromVector(
    const std::vector<int>& image, const std::vector<float>& disp,
    const int image_width, const int image_height, const int disp_width,
    const int disp_height, const float focal, const float disp_mult,
    const int depth_cut_off, const int side_cut_off, const float height_cut_off,
    const float inlier_dist, const int plane_angle,
    std::vector<float>* output) {
  Mat image_mat, disp_mat;
  pcl::PointCloud<pcl::PointXYZRGBA> cloud;

  VectorToMatColor(image, image_width, image_height, 3, &image_mat);
  VectorToMatFloat(disp, disp_width, disp_height, 1, &disp_mat);

  GetFilteredCloud(image_mat, disp_mat, focal, disp_mult, depth_cut_off,
                   side_cut_off, height_cut_off, inlier_dist, plane_angle,
                   &cloud);

  // PointCloudToVectorFloat checks that output is not null.
  PointCloudToVectorFloat(cloud, output);
}

void PCOps::GenerateInstanceImageFromVector(
    const std::vector<int>& image, const std::vector<float>& disp,
    const int image_width, const int image_height, const int disp_width,
    const int disp_height, const float focal, const float disp_mult,
    const int depth_cut_off, const int side_cut_off, const float height_cut_off,
    const float inlier_dist, const int plane_angle, const bool return_color,
    std::vector<int>* output) {
  Mat image_mat, disp_mat, out_mat;

  VectorToMatColor(image, image_width, image_height, 3, &image_mat);
  VectorToMatFloat(disp, disp_width, disp_height, 1, &disp_mat);

  GenerateInstanceImage(image_mat, disp_mat, focal, disp_mult, depth_cut_off,
                        side_cut_off, height_cut_off, inlier_dist, plane_angle,
                        return_color, &out_mat);

  if (!out_mat.empty()) {
    if (return_color) {
      MatToVectorColor(out_mat, output);
    } else {
      MatToVectorInt(out_mat, output);
    }
  }
}

}  // namespace point_cloud_ops
