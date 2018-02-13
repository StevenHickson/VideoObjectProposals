#ifndef CONVERSIONS_H_
#define CONVERSIONS_H_

#include "opencv2/core/core.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

int Clamp(int val, int min, int max);

int BoundValue(const int value, const int min, const int max);

void VectorToMatColor(const std::vector<int>& input, const int width,
                      const int height, const int nChannels, cv::Mat* output);

void VectorToMatFloat(const std::vector<float>& input, const int width,
                      const int height, const int nChannels, cv::Mat* output);

void VectorToMatInt(const std::vector<int>& input, const int width,
                      const int height, const int nChannels, cv::Mat* output);

void MatToVectorFloat(const cv::Mat& input, const int n_channels,
                      std::vector<float>* output);

void MatToVectorColor(const cv::Mat& input, std::vector<int>* output);

void MatToVectorInt(const cv::Mat& input, std::vector<int>* output);

void PointCloudToVectorFloat(const pcl::PointCloud<pcl::PointXYZRGBA>& pc,
                             std::vector<float>* output);

void PointCloudToVectorFloat(const pcl::PointCloud<pcl::Normal>& pc,
                             std::vector<float>* output);

void PointCloudToVectorInt(const pcl::PointCloud<pcl::PointXYZI>& pc,
                           std::vector<int>* output);

void PointCloudToVectorColor(const pcl::PointCloud<pcl::PointXYZRGBA>& pc,
                             std::vector<int>* output);

#endif // CONVERSIONS_H_
