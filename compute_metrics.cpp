// Computes metrics using the intersection over the union.
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
#include "conversions.h"
#include <fstream>
#include "opencv2/core/core.hpp"
#include <4D_Segmentation.h>

DEFINE_string(file_list, "", "The list of files to parse.");

using namespace cv;
using namespace std;

void CalcIoU(const Mat &inst, const Mat &gt, const Mat &label, std::map<int,int> *recallCount) {
  // we need to go through each instance in the gt and check for iou
  std::map<int,int> positiveCount, totalCount;
  std::map<std::pair<int,int>, std::vector<int> > gt_inst_inter;
  Mat_<uchar>::const_iterator pG = gt.begin<uchar>();
  Mat_<uchar>::const_iterator pL = label.begin<uchar>();
  Mat_<uchar>::const_iterator pI = inst.begin<uchar>();
  while(pG != gt.end<uchar>()) {
    // For every different gt instance, we need to calculate the intersection and union for every instance
    // We can make this faster by only considering when the intersections are nonzero
    if(*pG != 0 && *pI != 0) {
      //if(gt_inst_iter.find(*pG) != gt_inst_iter.end())
      gt_inst_inter[std::pair<int,int>(*pG,*pL)].push_back(*pI);
    }
    ++pG; ++pI; ++pL;
  }
}

int main(int argc, char* argv[]) {
  std::map<int,int> recallCount;

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ifstream in(FLAGS_file_list);
  if(!in)
  {
    cout << "Cannot open file list! " << FLAGS_file_list  << endl;
    return 1;
  }

  string line;
  while(getline(in, line)) {
    string inst_name, gt_name, label_name;
    cv::Mat inst, gt, label;
    inst_name = regex_replace(line, regex("png"), "inst.png");
    inst = imread(inst_name, CV_LOAD_IMAGE_GRAYSCALE);
    if (inst.empty()) {
      cout << "Error, couldn't load inst" << endl;
      break;
    }
    gt_name = regex_replace(line, regex("leftImg8bit_"), "gtFine_instanceIds_");
    gt_name = regex_replace(gt_name, regex("leftImg8bit"), "gtFine");
    gt = imread(gt_name, CV_LOAD_IMAGE_GRAYSCALE);
    if (gt.empty()) {
      cout << "Error, couldn't load gt" << endl;
      break;
    }
    label_name = regex_replace(line, regex("leftImg8bit_"), "gtFine_labelIds_");
    label_name = regex_replace(label_name, regex("leftImg8bit"), "gtFine");
    label = imread(label_name, CV_LOAD_IMAGE_GRAYSCALE);
    if (label.empty()) {
      cout << "Error, couldn't load label" << endl;
      break;
    }
    
    //Let's calculate the intersection and union for each segment
    CalcIoU(inst, gt, label, &recallCount);
  }
  return 0;
}
