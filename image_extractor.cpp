#include "image_extractor.h"
#include "conversions.h"

#include <math.h>
#include <stdlib.h>
#include <algorithm>

//#include "util/random/mt_random.h"

#include "city_scape_info.h"

using cityscape::GetTrainId;
using cityscape::GetTrainIdFullUnsup;
using namespace cv;

namespace {

void ObjectMaptoVector(const std::map<int, unsup_clustering::ObjectInfo>& input,
                       std::vector<unsup_clustering::ObjectInfo>* output) {
  output->resize(input.size());
  std::map<int, unsup_clustering::ObjectInfo>::const_iterator p_input =
      input.begin();
  std::vector<unsup_clustering::ObjectInfo>::iterator p_output =
      output->begin();
  while (p_input != input.end()) {
    *p_output++ = p_input->second;
    p_input++;
  }
}

void ObjectVectorToMap(const std::vector<unsup_clustering::ObjectInfo>& input,
                       std::map<int, unsup_clustering::ObjectInfo>* output) {
  std::vector<unsup_clustering::ObjectInfo>::const_iterator pI = input.begin();
  int i = 0;
  while (pI != input.end()) {
    (*output)[i] = *pI;
    pI++;
    i++;
  }
}

bool DoesPatchOverlap(int targetXMin, int targetYMin, int targetXMax,
                      int targetYMax, int patchXMin, int patchYMin,
                      int patchXMax, int patchYMax) {
  // If one rectangle is on left side of other then we don't overlap.
  if (targetXMin >= patchXMax || patchXMin >= targetXMax) {
    return false;
  }
  // If one rectangle is above the other then we don't overlap.
  if (targetYMin <= patchYMax || patchYMin <= targetYMax) {
    return false;
  }
  return true;
}

int UniformRandom(const int range_start, const int range_end) {
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_int_distribution<int>  randGen(range_start, range_end);
  return randGen(generator);
}

}  // namespace

namespace unsup_clustering {

void Extractor::ExtractInstanceData(const Mat& image,
                                    const Mat& labels,
                                    const Mat& instances,
                                    const bool fully_unsup,
                                    std::map<int, ObjectInfo>* instanceMap) {
  // Go through the instances, check the label for each instance, if it is in
  // label_map, then we care about it and it is label 1, extract the box and
  // add the appropriate label.

  // We use a map here to contain the bounding box, label, and instance info
  // because the instances aren't in consecutive order.
  //const uint8* instanceData = instances.pixel_data();
  //const uint8* labelData = labels.pixel_data();
  cv::Mat_<uchar>::const_iterator instanceData = instances.begin<uchar>();
  cv::Mat_<uchar>::const_iterator labelData = labels.begin<uchar>();
  // Here we loop over the image looking for every available instance id.
  for (int j = 0; j < instances.rows; j++) {
    for (int i = 0; i < instances.cols; i++, labelData++, instanceData++) {
      if (*instanceData != 0) {
        std::map<int, ObjectInfo>::iterator pI =
            instanceMap->find(*instanceData);
        if (pI != instanceMap->end()) {
          // The instance exists and we need to update the data
          if (pI->second.x_min_ > i) pI->second.x_min_ = i;
          if (pI->second.x_max_ < i) pI->second.x_max_ = i;
          if (pI->second.y_min_ > j) pI->second.y_min_ = j;
          if (pI->second.y_max_ < j) pI->second.y_max_ = j;
        } else {
          // New instance and we should insert it in the map
          ObjectInfo newInfo;
          if (fully_unsup) {
            newInfo.label_ = GetTrainIdFullUnsup(*labelData);
          } else {
            newInfo.label_ = GetTrainId(*labelData);
          }
          newInfo.x_min_ = newInfo.x_max_ = i;
          newInfo.y_min_ = newInfo.y_max_ = j;
          newInfo.instance_ = *instanceData;
          (*instanceMap)[*instanceData] = newInfo;
        }
      }
    }
  }
}

void Extractor::CreateNegativePatches(const int input_width,
                                      const int input_height,
                                      const int patch_min_size,
                                      const int patch_max_size,
                                      const int numPatches,
                                      const std::vector<int>& positiveLabels,
                                      const std::map<int, ObjectInfo>& inputMap,
                                      std::map<int, ObjectInfo>* outputMap) {
  // We need to grab random background patches of different sizes.
  // They shouldn't contain any positiveLabels though.
  int pCount = 0;
  int instanceNum;
  // We need to check for the rare case when there are no positive instances.
  if (inputMap.empty())
    instanceNum = 1;
  else
    instanceNum = inputMap.rbegin()->first + 1;
  int patchDiff = patch_max_size - patch_min_size;
  int safeWidth = input_width - patch_max_size - 1;
  int safeHeight = input_height - patch_max_size - 1;
  //MTRandom randGen;
  // Copy the inputMap into the outputMap.
  *outputMap = inputMap;
  while (pCount < numPatches) {
    // Grab a random point and a random width, height patch.
    int randMinX = UniformRandom(0, safeWidth);
    int randMinY = UniformRandom(0, safeHeight);
    int randSize = UniformRandom(0, patchDiff) + patch_min_size;
    int randMaxX = randSize + randMinX;
    int randMaxY = randSize + randMinY;
    // Let's check to make sure the patch doesn't overlap any positiveLabels.
    bool overlapsPositive = false;
    std::map<int, ObjectInfo>::const_iterator pI = inputMap.begin();
    while (pI != inputMap.end() && !overlapsPositive) {
      int xMax = pI->second.x_min_ + pI->second.x_max_;
      int yMax = pI->second.y_min_ + pI->second.y_max_;
      if ((std::find(positiveLabels.begin(), positiveLabels.end(),
                     pI->second.label_) != positiveLabels.end()) &&
          DoesPatchOverlap(pI->second.x_min_, pI->second.y_min_, xMax, yMax,
                           randMinX, randMinY, randMaxX, randMaxY)) {
        overlapsPositive = true;
        break;
      }
      ++pI;
    }
    if (!overlapsPositive) {
      pCount++;
      instanceNum++;
      ObjectInfo newObject =
          ObjectInfo(randMinX, randMinY, randSize, randSize, 0, 0);
      // We index negatively in the map in order to distinguish the background
      // samples from the foreground samples later.
      (*outputMap)[instanceNum] = newObject;
      //std::cout << "Negative sample: " << randMinX << "," << randMinY << "," << randSize << std::endl;
    }
  }
}

void Extractor::CreateDataFromInstances(
    const int input_width, const int input_height, const int output_width,
    const int output_height, const int min_size, const float max_aspect_ratio,
    const bool keep_aspect_ratio, const std::map<int, ObjectInfo>& instanceMap,
    std::vector<ObjectInfo>* croppedList) {
  // Go through each instance, save the original label, mapped label, and seg.
  std::map<int, ObjectInfo>::const_iterator pI = instanceMap.begin();
  float aspect_ratio =
      static_cast<float>(output_width) / static_cast<float>(output_height);

  while (pI != instanceMap.end()) {
    // Let's extract the image from the bounding box.
    int newWidth = pI->second.x_max_ - pI->second.x_min_;
    int newHeight = pI->second.y_max_ - pI->second.y_min_;
    if (newWidth < 0) {
      newWidth = 0;
    }
    if (newHeight < 0) {
      newHeight = 0;
    }

    float segment_aspect =
        static_cast<float>(newWidth) / static_cast<float>(newHeight);
    int newX = pI->second.x_min_, newY = pI->second.y_min_;
    int halfWidth = newWidth / 2;
    int halfHeight = newHeight / 2;
    if (keep_aspect_ratio) {
      int centerX = halfWidth + pI->second.x_min_;
      int centerY = halfHeight + pI->second.y_min_;
      newWidth =
          static_cast<int>(fmax(static_cast<float>(newWidth),
                                static_cast<float>(newHeight) * aspect_ratio));
      newHeight =
          static_cast<int>(fmax(static_cast<float>(newHeight),
                                static_cast<float>(newWidth) / aspect_ratio));
      halfWidth = newWidth / 2;
      halfHeight = newHeight / 2;
      newX = max(min(centerX - halfWidth, input_width), 0);
      newY = max(min(centerY - halfHeight, input_height), 0);
    }
    if (newWidth + newX >= input_width) {
      newWidth = input_width - newX - 1;
    }
    if (newHeight + newY >= input_height) {
      newHeight = input_height - newY - 1;
    }

    // Let's be safe about this and make sure our instances aren't too small.
    // Let's also make sure our aspect_ratio isn't above the max.
    if (newWidth > min_size && newHeight > min_size &&
        (segment_aspect < max_aspect_ratio || max_aspect_ratio == 0)) {
      // Note that we set newWidth and newHeight as max_x and min_x due to the
      // fact that our crop expects left corner position and width, height
      // rather than rectangle coordinates.
      ObjectInfo newObject =
          ObjectInfo(newX, newY, newWidth, newHeight,
                     static_cast<int>(pI->second.label_), pI->first);
      croppedList->push_back(newObject);
    } else {
      //std::cout << "Throwing out object with following info: " << newX << "," << newY << "," << newWidth << "," << newHeight << "," << pI->second.label_ << "," << pI->first << std::endl;
    }
    ++pI;
  }
}

void Extractor::EqualizeDataInstances(
    const int input_width, const int input_height, const int jitter,
    const std::vector<ObjectInfo>& instanceMap,
    std::vector<ObjectInfo>* croppedList) {
  // Go through each instance, save the original label, mapped label, and seg.
  std::vector<ObjectInfo>::const_iterator pI = instanceMap.begin();

  //MTRandom randGen;
  while (pI != instanceMap.end()) {
    // Let's go ahead and copy the data
    croppedList->push_back(*pI);
    // Let's see if it's one of our special cases
    int mult = cityscape::GetMult(pI->label_);
    // jitter is +/- so we will make a double jitter here.
    int doubJitter = jitter * 2;
    for (int i = 1; i < mult; i++) {
      int newX, newY, newWidth, newHeight;
      newX = UniformRandom(0, doubJitter) - jitter + pI->x_min_;
      newY = UniformRandom(0, doubJitter) - jitter + pI->y_min_;
      newWidth = UniformRandom(0, doubJitter) - jitter + pI->x_max_;
      newHeight = UniformRandom(0, doubJitter) - jitter + pI->y_max_;

      // Now let's make sure they are valid numbers.
      newX = BoundValue(newX, 0, input_width);
      newY = BoundValue(newY, 0, input_height);
      newWidth = BoundValue(newWidth, 0, input_width - newX);
      newHeight = BoundValue(newHeight, 0, input_height - newY);
      ObjectInfo newObject =
          ObjectInfo(newX, newY, newWidth, newHeight,
                     static_cast<int>(pI->label_), pI->instance_);
      croppedList->push_back(newObject);
    }
    ++pI;
  }
}

void Extractor::ExtractInstanceDataFromVector(
    const std::vector<int>& image, const std::vector<int>& labels,
    const std::vector<int>& instances, const int width, const int height,
    const int nChannels, const bool fully_unsup,
    std::vector<ObjectInfo>* instanceList) {
  // Convert our vector<int>s to Mats
  Mat imageRaw, labelsRaw, instancesRaw;
  VectorToMatColor(image, width, height, nChannels, &imageRaw);
  VectorToMatInt(labels, width, height, nChannels, &labelsRaw);
  VectorToMatInt(instances, width, height, nChannels, &instancesRaw);

  std::map<int, ObjectInfo> instanceMap;
  ExtractInstanceData(imageRaw, labelsRaw, instancesRaw, fully_unsup,
                      &instanceMap);

  // convert std::map to std::vector
  ObjectMaptoVector(instanceMap, instanceList);
}

void Extractor::CreateNegativePatchesFromVector(
    const int input_width, const int input_height, const int patch_min_size,
    const int patch_max_size, const int numPatches,
    const std::vector<int>& positiveLabels,
    const std::vector<ObjectInfo>& inputMap,
    std::vector<ObjectInfo>* outputMap) {
  std::map<int, ObjectInfo> inputVector, outputVector;
  ObjectVectorToMap(inputMap, &inputVector);
  CreateNegativePatches(input_width, input_height, patch_min_size,
                        patch_max_size, numPatches, positiveLabels, inputVector,
                        &outputVector);
  ObjectMaptoVector(outputVector, outputMap);
}

void Extractor::CreateDataFromInstancesFromVector(
    const int input_width, const int input_height, const int output_width,
    const int output_height, const int min_size, const float max_aspect_ratio,
    const std::vector<ObjectInfo>& instanceList,
    std::vector<ObjectInfo>* croppedList) {
  std::map<int, ObjectInfo> instanceMap;
  ObjectVectorToMap(instanceList, &instanceMap);
  CreateDataFromInstances(input_width, input_height, output_width,
                          output_height, min_size, max_aspect_ratio, true,
                          instanceMap, croppedList);
}

}  // namespace unsup_clustering
