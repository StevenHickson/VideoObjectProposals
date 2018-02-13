#ifndef IMAGE_EXTRACTOR_H_
#define IMAGE_EXTRACTOR_H_
#include <map>
#include <vector>

#include "opencv2/core/core.hpp"

namespace unsup_clustering {

class ObjectInfo {
 public:
  ObjectInfo() {}
  ObjectInfo(int xMin, int yMin, int xMax, int yMax, int _label, int _instance)
      : x_min_(xMin),
        y_min_(yMin),
        x_max_(xMax),
        y_max_(yMax),
        label_(_label),
        instance_(_instance) {}
  int x_min_, y_min_, x_max_, y_max_;
  int label_;
  int instance_;
};

class Extractor {
 public:
  Extractor() {}

  // Here we extract instances from the image and label them appropriately. The
  // output instanceMap is a map of ObjectInfo corresponding to each patches'
  // bounding box, instance id, and label.
  void ExtractInstanceData(const cv::Mat& image,
                           const cv::Mat& labels,
                           const cv::Mat& instances,
                           const bool fully_unsup,
                           std::map<int, ObjectInfo>* instanceMap);

  // This function creates background sample patches that don't overlap with the
  // positive sample patches.
  void CreateNegativePatches(const int input_width, const int input_height,
                             const int patch_min_size, const int patch_max_size,
                             const int numPatches,
                             const std::vector<int>& positiveLabels,
                             const std::map<int, ObjectInfo>& inputMap,
                             std::map<int, ObjectInfo>* outputMap);

  // Here we modify both our negative and positive sample patches, correcting
  // for aspect ratio and formatting them properly.
  void CreateDataFromInstances(const int input_width, const int input_height,
                               const int output_width, const int output_height,
                               const int min_size, const float max_aspect_ratio,
                               const bool keep_aspect_ratio,
                               const std::map<int, ObjectInfo>& instanceMap,
                               std::vector<ObjectInfo>* croppedList);

  // Here we equalize the data instances so there are an equal # per class.
  void EqualizeDataInstances(const int input_width, const int input_height,
                             const int jitter,
                             const std::vector<ObjectInfo>& instanceMap,
                             std::vector<ObjectInfo>* croppedList);

  // The following FromVector functions are for debugging and using with Clif.
  void ExtractInstanceDataFromVector(const std::vector<int>& image,
                                     const std::vector<int>& labels,
                                     const std::vector<int>& instances,
                                     const int width, const int height,
                                     const int nChannels,
                                     const bool fully_unsup,
                                     std::vector<ObjectInfo>* instanceList);

  void CreateNegativePatchesFromVector(const int input_width,
                                       const int input_height,
                                       const int patch_min_size,
                                       const int patch_max_size,
                                       const int numPatches,
                                       const std::vector<int>& positiveLabels,
                                       const std::vector<ObjectInfo>& inputMap,
                                       std::vector<ObjectInfo>* outputMap);

  void CreateDataFromInstancesFromVector(
      const int input_width, const int input_height, const int output_width,
      const int output_height, const int min_size, const float max_aspect_ratio,
      const std::vector<ObjectInfo>& instanceList,
      std::vector<ObjectInfo>* croppedList);
};

}  // namespace unsup_clustering
#endif  // IMAGE_EXTRACTOR_H_

