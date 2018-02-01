#ifndef POINT_CLOUD_OPS_H_
#define POINT_CLOUD_OPS_H_

#include "opencv2/core/core.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <4D_Segmentation.h>

namespace point_cloud_ops {

class PCOps {
 private:
  SegmentationOptions* options_;
  RGBDTSegmentation* segments_;

 public:
  PCOps() : options_(NULL), segments_(NULL) {}

  // This function computes Farneback optical flow using OpenCV.
  void ComputeOpticalFlow(const cv::Mat& past, const cv::Mat& current,
                          cv::Mat* output);

  // This extends Farneback optical flow to 3D point clouds.
  void ComputeOpticalFlow3D(
      const cv::Mat& past, const cv::Mat& current,
      const pcl::PointCloud<pcl::PointXYZRGBA>& past_cloud,
      const pcl::PointCloud<pcl::PointXYZRGBA>& current_cloud,
      const float focal_length, pcl::PointCloud<pcl::Normal>* output);

  // This function sets up the segmentation options for the Segmentation_4D lib.
  void SetSegmentationOptions();

  // This function generates the segments given a cloud.
  void GetSegments(const pcl::PointCloud<pcl::PointXYZRGBA>& cloud,
                   pcl::PointCloud<pcl::PointXYZI>* labels,
                   pcl::PointCloud<pcl::PointXYZRGBA>* labels_color);

  // Here we convert disparity to depth using camera parameters.
  void DisparityToDepth(const cv::Mat& input_disparity, const int new_width,
                        const int new_height, const float focal_point,
                        const float disp_mult, cv::Mat* output_depth);

  // This function trims the output_cloud from input_cloud by removing the
  // ground plane and cutting off any pixel too far away, too high, or too close
  // to the sides.
  void TrimDepth(const pcl::PointCloud<pcl::PointXYZRGBA>& input_cloud,
                 const pcl::PointIndices& ground_plane,
                 const float depth_cut_off, const int side_cut_off,
                 const float height_cut_off,
                 pcl::PointCloud<pcl::PointXYZRGBA>* output_cloud);

  // Here we create a point cloud given an input image,depth pair.
  void CreatePointCloud(const cv::Mat& input_image,
                        const cv::Mat& input_depth,
                        pcl::PointCloud<pcl::PointXYZRGBA>* cloud,
                        const float focal = 1000);

  // This creates the fully filtered cloud from input image and disparity.
  // It calls DisparityToDepth, CreatePointCloud, ComputeGroundPlane, and
  // TrimDepth.
  void GetFilteredCloud(const cv::Mat& input_image, const cv::Mat& input_disp,
                        const float focal, const float disp_mult,
                        const int depth_cut_off, const int side_cut_off,
                        const float height_cut_off, const float inlier_dist,
                        const int plane_angle,
                        pcl::PointCloud<pcl::PointXYZRGBA>* output_cloud);

  // This is a helper function to get a floating point Mat from a point cloud.
  void GetMatFromCloud(const pcl::PointCloud<pcl::PointXYZRGBA>& cloud,
                       cv::Mat* image);

  // This is a helper function to get a RGB Mat from a point cloud.
  void GetColoredMatFromCloud(const pcl::PointCloud<pcl::PointXYZRGBA>& cloud,
                              cv::Mat* image);

  // This is a helper function to get a int labeled Mat from a point cloud.
  void GetLabeledMatFromCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud,
                              cv::Mat* image);

  // This computes the ground plane using a number of methods in PCL.
  void ComputeGroundPlane(const pcl::PointCloud<pcl::PointXYZRGBA>& input_cloud,
                          const float inlier_dist, const int plane_angle,
                          const int method, const int plane_selection,
                          pcl::PointIndices* inliers);

  // This gets the inliers from the ground plane and height cut off.
  void GetCutOffInliers(const pcl::PointCloud<pcl::PointXYZRGBA>& input_cloud,
                        const float height_cut_off,
                        const std::vector<int>& ground_plane,
                        const bool above_plane, std::vector<int>* indices);

  // This generates the full instance image from the given image, disp pair.
  // It calls GetFilteredCloud and converts things appropriately.
  void GenerateInstanceImage(const cv::Mat& input_image,
                             const cv::Mat& input_disp, const float focal,
                             const float disp_mult, const int depth_cut_off,
                             const int side_cut_off, const float height_cut_off,
                             const float inlier_dist, const int plane_angle,
                             const bool return_color, cv::Mat* output_image);

  // This does the same as above but makes the background class 0 for ease.
  void GenerateZeroBackgroundInstances(
      const cv::Mat& input_image, const cv::Mat& input_disp,
      const float focal, const float disp_mult, const int depth_cut_off,
      const int side_cut_off, const float height_cut_off,
      const float inlier_dist, const int plane_angle, cv::Mat* output_image);

  // The following FromVector functions are for debugging and using with Clif.

  void ComputeOpticalFlowFromVector(const std::vector<int>& past_image,
                                    const std::vector<int>& current_image,
                                    const int width, const int height,
                                    std::vector<float>* flow);

  void ComputeOpticalFlow3DFromVector(const std::vector<int>& past_image,
                                      const std::vector<float>& past_depth,
                                      const std::vector<int>& current_image,
                                      const std::vector<float>& current_depth,
                                      const int width, const int height,
                                      const float focal,
                                      std::vector<float>* flow);

  void GetGroundPlaneFromVector(const std::vector<int>& image,
                                const std::vector<float>& depth,
                                const int width, const int height,
                                const int nChannels, const float focal,
                                const float inlier_dist, const int plane_angle,
                                const int method, const int plane_selection,
                                std::vector<int>* indices);

  void CreatePointCloudFromVector(const std::vector<int>& image,
                                  const std::vector<float>& depth,
                                  const int width, const int height,
                                  const float focal,
                                  std::vector<float>* output);

  void GetSegmentsFromVector(const std::vector<int>& image,
                             const std::vector<float>& depth, const int width,
                             const int height, const float focal,
                             const bool return_color, std::vector<int>* output);

  void GetCutOffInliersFromVector(const std::vector<int>& image,
                                  const std::vector<float>& depth,
                                  const int width, const int height,
                                  const float focal, const float height_cut_off,
                                  const std::vector<int>& ground_plane,
                                  const bool above_plane,
                                  std::vector<int>* output);

  void GetFilteredCloudFromVector(
      const std::vector<int>& image, const std::vector<float>& disp,
      const int image_width, const int image_height, const int disp_width,
      const int disp_height, const float focal, const float disp_mult,
      const int depth_cut_off, const int side_cut_off,
      const float height_cut_off, const float inlier_dist,
      const int plane_angle, std::vector<float>* output);

  void GenerateInstanceImageFromVector(
      const std::vector<int>& image, const std::vector<float>& disp,
      const int image_width, const int image_height, const int disp_width,
      const int disp_height, const float focal, const float disp_mult,
      const int depth_cut_off, const int side_cut_off,
      const float height_cut_off, const float inlier_dist,
      const int plane_angle, const bool return_color, std::vector<int>* output);

  // Loads the depth files generated with monodepth.
  void LoadNPY(const std::string& filename, int* width, int* height,
               std::vector<float>* output);
};

}  // namespace point_cloud_ops

#endif  // POINT_CLOUD_OPS_H_
