#include "conversions.h"
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

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

int Clamp(int val, int min, int max) {
  if (val > max) return max;
  if (val < min) return min;
  return val;
}

int BoundValue(const int value, const int min, const int max) {
  if (value < min) {
    return min;
  }
  if (value >= max) {
    return max - 1;
  }
  return value;
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

void VectorToMatInt(const std::vector<int>& input, const int width,
                      const int height, const int nChannels, Mat* output) {
  if (output == NULL) {
    std::cout << "Output Mat cannot be NULL in VectorToMatFloat" << std::endl;
    return;
  }
  *output = Mat(height, width, CV_32SC1);
  typename Mat_<int>::iterator pO = output->begin<int>();
  typename std::vector<int>::const_iterator p = input.begin();
  while (p != input.end()) {
    *pO++ = static_cast<int>(*p++);
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

