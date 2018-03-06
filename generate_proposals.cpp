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


DEFINE_string(file_list, "", "The list of files to parse.");

DEFINE_int32(min_size, 64,
             "The minimum width and height for an instance mask.");

DEFINE_int32(output_height, 256,
             "The size of the network input window. "
             "All output images will be resized to that before saving.");

DEFINE_int32(output_width, 512,
             "The size of the network input window. "
             "All output images will be resized to that before saving.");

DEFINE_enum(
    string, object_subset, "NORMAL",
    "Which split of true labels to use as samples. Allowed modes:\n  NORMAL: "
    "This is the full cityscapes split.\n  SMALL: This is just car, bicycle, "
    "and person.\n  UNSUP: This is the unsupervised version.",
    "UNSUP", "SMALL", "UNSUP");

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

DEFINE_int32(num_patches, 20, "The number of negative patches to generate.");

DEFINE_int32(jitter, 0,
             "The amount (+/-) to jitter the bounding boxes of samples "
             "generated from the image.");

DEFINE_bool(use_optical_flow, false,
            "If true, will use optical flow to prune from positive samples.");

DEFINE_bool(
    use_optical_flow_3d, false,
    "If true, will use 3d optical flow to prune from positive samples.");

DEFINE_double(optical_flow_dist, 10, "The optical flow distance.");

using namespace cv;
using namespace std;


void GetOpticalFlow(const Mat& past_image, const Mat& current_image,
                    Mat* flow) {
  point_cloud_ops::PCOps pcops;
  pcops.ComputeOpticalFlow(past_image, current_image, flow);
}

void CalcBackgroundFlow(const Mat& instange_image, const Mat& flow,
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
  Scalar mean = mean(roiMat);
  if (roiMat.empty()) {
    cout << "Couldn't calc flow dist" << endl;
    return 0;
  }
  return std::abs(mean[0] - background_flow[0]) +
         std::abs(mean[1] - background_flow[1]);
}

int main(int argc, char* argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ifstream in(FLAGS_file_list);
  if(!in)
  {
    cout << "Cannot open file list! " << FLAGS_file_list  << endl;
    return 1;
  }

  string line;
  while(getline(in, line)) {
    
  }
  return 0;
}
