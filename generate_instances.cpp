// Genereates instance labels using the proposal method.
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

DEFINE_string(file_list, "", "The list of files to parse.");

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

void ConvertTo8Bit(const cv::Mat& input_image, cv::Mat* output_image) {
  if (output_image == nullptr) {
    cout << "Output Mat cannot be NULL in ConvertTo8Bit" << endl;
    return;
  }
  *output_image = cv::Mat(input_image.rows, input_image.cols, CV_8UC1);
  std::map<int, int> instanceMapper;
  int count = 1;
  cv::Mat_<int>::const_iterator pI = input_image.begin<int>();
  cv::Mat_<char>::iterator pO = output_image->begin<char>();
  while (pI != input_image.end<int>()) {
    if (*pI == 0) {
      *pO = 0;
    } else {
      std::map<int, int>::iterator pM = instanceMapper.find(*pI);
      if (pM != instanceMapper.end()) {
        // The instance already exists
        *pO = pM->second;
      } else {
        *pO = count;
        instanceMapper[*pI] = count;
        count++;
      }
    }
    pI++;
    pO++;
  }
}

bool ReadFloatImage(const string& uid, const int width, const int height,
                    cv::Mat* image) {
  *image = cv::Mat(height, width, CV_32F);
  streampos size;
  char *bytes;

  ifstream file (uid, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    bytes = new char [size];
    file.seekg (0, ios::beg);
    file.read (bytes, size);
    file.close();

    memcpy(&*image->begin<float>(), bytes,
           height * width * sizeof(float));
    delete[] bytes;
  } else {
	printf("Error, can't read: %s\n", uid.c_str());
	return false;
  }

  return true;
}

int main(int argc, char* argv[]) {

  ifstream in(FLAGS_file_list);
  if(!in)
  {
    cout << "Cannot open file list!" << endl;
    return 1;
  }

  string line;
  while(getline(in, line)) {
  string disp_name, inst_name, label_name;
  cv::Mat image, disp, instance_img, instance_gray, label_img;
  disp_name = regex_replace(line, regex("png"), "flt");
  if (!ReadFloatImage(disp_name, FLAGS_image_width, FLAGS_image_height, &disp)) {
          printf("Error, couldn't load flt image\n");
          break;
  }
  point_cloud_ops::PCOps pcops;
  pcops.SetSegmentationOptions();
  pcops.GenerateZeroBackgroundInstances(
      image, disp, FLAGS_focal_length, FLAGS_disp_mult,
      static_cast<int>(FLAGS_depth_cut_off), FLAGS_side_cut_off,
      FLAGS_height_cut_off, FLAGS_inlier_dist, FLAGS_plane_angle,
      &instance_img);

  ConvertTo8Bit(instance_img, &instance_gray);
  cv::threshold(instance_gray, label_img, 1, 1, CV_THRESH_TRUNC);
  inst_name = regex_replace(line, regex("png"), "inst.png");
  label_name = regex_replace(line, regex("png"), "label.png");
  imwrite(inst_name, instance_gray);
  imwrite(label_name, label_img);
  }
  return 0;
}
