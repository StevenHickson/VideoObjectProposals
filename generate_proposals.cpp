// Genereates proposals from instances.
//
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
#include "image_extractor.h"
#include "conversions.h"
#include "point_cloud_ops.h"
#include "city_scape_info.h"


DEFINE_string(file_list, "", "The list of files to parse.");

DEFINE_string(output_folder, "proposals", "The output folder name");

DEFINE_int32(min_size, 64,
             "The minimum width and height for an instance mask.");

DEFINE_int32(output_height, 256,
             "The size of the network input window. "
             "All output images will be resized to that before saving.");

DEFINE_int32(output_width, 512,
             "The size of the network input window. "
             "All output images will be resized to that before saving.");

DEFINE_int32(object_subset, 0, "0 == normal, 1 == small, 2 == unsupervised.");

DEFINE_bool(background_patches, false,
            "If true, will generate random background patches.");

DEFINE_double(
    max_aspect_ratio, 6,
    "The aspect ratio (width/height) above which to remove proposals.");

DEFINE_bool(keep_aspect_ratio, true,
            "If true, will keep natural aspect ratio in patches otherwise it "
            "will stretch the image patch.");

DEFINE_int32(patch_min_size, 64,
             "The minimum width and height for a negative patch.");

DEFINE_int32(patch_max_size, 64,
             "The maximum width and height for a negative patch.");

DEFINE_int32(num_patches, 10, "The number of negative patches to generate.");

DEFINE_int32(jitter, 0,
             "The amount (+/-) to jitter the bounding boxes of samples "
             "generated from the image.");

DEFINE_bool(use_optical_flow, false,
            "If true, will use optical flow to prune from positive samples.");

DEFINE_bool(
    use_optical_flow_3d, false,
    "If true, will use 3d optical flow to prune from positive samples.");

DEFINE_double(optical_flow_dist, 10, "The optical flow distance.");

DEFINE_bool(infer_labels, false, "Whether to infer the gt label.");

using namespace cv;
using namespace std;
using namespace unsup_clustering;
using namespace cityscape;


void GetOpticalFlow(const Mat& past_image, const Mat& current_image,
                    Mat* flow) {
  point_cloud_ops::PCOps pcops;
  pcops.ComputeOpticalFlow(past_image, current_image, flow);
}

void CalcBackgroundFlow(const Mat& instance_image, const Mat& flow,
                        Vec2f* background_flow) {
  Mat_<Vec2f>::const_iterator pF = flow.begin<Vec2f>();
  Mat_<uchar>::const_iterator pI = instance_image.begin<uchar>();
  *background_flow = Vec2f(0, 0);
  int count = 0;
  while (pF != flow.end<Vec2f>()) {
    if (*pI == 0) {
      *background_flow += *pF;
      ++count;
    }
    ++pF; ++pI;
  }
  *background_flow /= count;
}

float CalcFlowDistance(const Mat& flow, const Rect& roi,
                       const Vec2f& background_flow) {
  Mat roiMat = flow(roi);
  Scalar mean = cv::mean(roiMat);
  if (roiMat.empty()) {
    cout << "Couldn't calc flow dist" << endl;
    return 0;
  }
  return std::abs(mean[0] - background_flow[0]) +
         std::abs(mean[1] - background_flow[1]);
}

int InferLabel(const Mat &label_img, const Rect &roi) {
  Mat label_crop = label_img(roi).clone();
  Mat_<uchar>::const_iterator p = label_crop.begin<uchar>();
  vector<int> counts(256);
  while (p != label_crop.end<uchar>()) {
    counts[*p++]++;
  }
  int max = 0, maxLoc = 0;
  vector<int>::const_iterator pV = counts.begin();
  int c = 0;
  while (pV != counts.end()) {
    if(*pV > max) {
      max = *pV;
      maxLoc = c;
    }
    ++c; ++pV;
  }
  return maxLoc;
}

class Generator {
  public:
    Generator() { current_count_ = 0; };
    void SaveData(const Mat &img, const Mat &label_image, const Rect &roi, const int label, const string &filename);
    void CreateExamplesFromInstances(
                  const Mat& input_image, const Mat& instance_images, const Mat& label_image,
                  const std::map<int, ObjectInfo>& instance_map, const string& filename);

    ofstream output_file_;
    int current_count_;
    Mat flow_;
    Mat background_flow_;
    Extractor extractor_;
};

void Generator::SaveData(const Mat &img, const Mat &label_image, const Rect &roi, const int label, const string &filename) {
  // Save the image in the proper directory, that is dictated by label
  int save_label;
  int actual_label = label;
  if (FLAGS_object_subset == 1) {
    save_label = GetTargetSubsetObjectIds(label);
  } else if (FLAGS_object_subset == 0) {
    save_label = GetTargetObjectIds(label);
  } else {
    if (FLAGS_infer_labels)
      actual_label = InferLabel(label_image, roi);
    if (label == 0)
      save_label = 0; 
    else if (label == 1 || label == 255)
      save_label = 1; 
    else
      cout << "Erroronous label!" << endl;
  }
  string folder;
  if (save_label == 0)
    folder = "background";
  else
    folder = "foreground";
  stringstream save_name;
  save_name << "/data/" << FLAGS_output_folder << "/" << folder << "/" << current_count_ << ".png";
  imwrite(save_name.str(), img);
  output_file_ << save_name.str() << "," << filename << "," << actual_label << "," << save_label << ",";
  output_file_ << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << "\n";
}


void Generator::CreateExamplesFromInstances(
    const Mat& input_image, const Mat& instance_images, const Mat& label_image,
    const std::map<int, ObjectInfo>& instance_map, const string& filename) {

  std::map<int, ObjectInfo> allInstances;
  std::vector<int> positiveLabels;
  if (FLAGS_object_subset == 0)
    positiveLabels = {11, 12, 13, 14, 15, 16, 17, 18};
  else if (FLAGS_object_subset == 1)
    positiveLabels = {11, 13, 18};
  else
    positiveLabels = {1};

  extractor_.CreateNegativePatches(input_image.cols, input_image.rows,
                                   FLAGS_patch_min_size, FLAGS_patch_max_size,
                                   FLAGS_num_patches, positiveLabels,
                                   instance_map, &allInstances);

  std::vector<ObjectInfo> croppedInstanceImages;
  extractor_.CreateDataFromInstances(
      input_image.cols, input_image.rows, FLAGS_output_width,
      FLAGS_output_height, FLAGS_min_size, FLAGS_max_aspect_ratio,
      FLAGS_keep_aspect_ratio, allInstances, &croppedInstanceImages);


  std::vector<ObjectInfo>::const_iterator pI = croppedInstanceImages.begin();

  while (pI != croppedInstanceImages.end()) {
    Rect roi(pI->x_min_, pI->y_min_, pI->x_max_, pI->y_max_);
    if (pI->label_ == 0 || !FLAGS_use_optical_flow ||
        CalcFlowDistance(flow_, roi, background_flow_) >= FLAGS_optical_flow_dist) {

      Mat cropped_image = input_image(roi).clone();
      Mat cropped_instance_image = instance_images(roi).clone();

      SaveData(cropped_image, label_image, roi, pI->label_, filename);
    }
    ++pI; ++current_count_;
  }
}

int main(int argc, char* argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ifstream in(FLAGS_file_list);
  if(!in)
  {
    cout << "Cannot open file list! " << FLAGS_file_list  << endl;
    return 1;
  }

  Generator gen;
  gen.output_file_.open("/data/" + FLAGS_output_folder + "/info.txt", ios::out);

  string line;
  while(getline(in, line)) {
    string disp_name, inst_name, label_name;
    Mat image, instance_img, label_img, gt_label_img;
    cout << "Parsing: " << line << endl;

    image = imread(line, CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
      cout << "Error, couldn't load image" << endl;
      break;
    }

    bool read_inst_16bit = false;
    if (FLAGS_object_subset == 2) {
      inst_name = regex_replace(line, regex("png"), "inst.png");
      label_name = regex_replace(line, regex("png"), "label.png");
    } else {
      inst_name = regex_replace(line, regex("_leftImg8bit"), "_gtFine_instanceIds");
      inst_name = regex_replace(inst_name, regex("leftImg8bit"), "gtFine");
      label_name = regex_replace(line, regex("_leftImg8bit"), "_gtFine_labelIds");
      label_name = regex_replace(label_name, regex("leftImg8bit"), "gtFine");
      read_inst_16bit = true;
    }

    if (read_inst_16bit)
      instance_img = imread(inst_name, CV_LOAD_IMAGE_ANYDEPTH);
    else
      instance_img = imread(inst_name, CV_LOAD_IMAGE_GRAYSCALE);
    if (instance_img.empty()) {
      cout << "Error, couldn't load instance" << endl;
      break;
    }
    
    label_img = imread(label_name, CV_LOAD_IMAGE_GRAYSCALE);
    if (label_img.empty()) {
      cout << "Error, couldn't load label" << endl;
      break;
    }

    if (FLAGS_infer_labels) {
      label_name = regex_replace(line, regex("_leftImg8bit"), "_gtFine_labelIds");
      label_name = regex_replace(label_name, regex("leftImg8bit"), "gtFine");
      gt_label_img = imread(label_name, CV_LOAD_IMAGE_GRAYSCALE);
      if (gt_label_img.empty()) {
        cout << "Error, couldn't load ground truth label" << endl;
        break;
      }
    }
    
    std::map<int, ObjectInfo> instanceMap;
    bool fully_unsup = false;
    if (FLAGS_object_subset == 2)
      fully_unsup = true;
    gen.extractor_.ExtractInstanceData(image, label_img, instance_img, fully_unsup, &instanceMap);
    gen.CreateExamplesFromInstances(image, instance_img, gt_label_img, instanceMap, line);
  }
  gen.output_file_.close();
  return 0;
}
