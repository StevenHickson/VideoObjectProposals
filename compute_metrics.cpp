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
#include <algorithm>
#include "opencv2/core/core.hpp"
#include <4D_Segmentation.h>

DEFINE_string(file_list, "", "The list of files to parse.");

DEFINE_double(iou_thresh, 0.5, "The iou_thresh.");

using namespace cv;
using namespace std;

float CalcIoUPerInst(const Mat &inst, const Mat &gt, const int inst_id, const int gt_id) {
  int interCount = 0, unionCount = 0;
  Mat_<unsigned short>::const_iterator pG = gt.begin<unsigned short>();
  Mat_<uchar>::const_iterator pI = inst.begin<uchar>();
  while(pG != gt.end<unsigned short>()) {
    // For every different gt instance, we need to calculate the intersection and union for every instance
    int pGNum = static_cast<int>(*pG);
    int pINum = static_cast<int>(*pI);
    if(pGNum == gt_id && pINum == inst_id)
      interCount++;
    if(pGNum == gt_id || pINum == inst_id)
      unionCount++;
    ++pG; ++pI;
  }
  return static_cast<float>(interCount) / static_cast<float>(unionCount);
}

void CalcIoU(const Mat &inst, const Mat &gt, const Mat &label, const float iou_thresh, std::vector<int> *positiveCountForLabel, std::vector<int> *totalCountForLabel) {
  // we need to go through each instance in the gt and check for iou
  std::vector< std::pair<int,int> > gt_inst_considered;
  std::vector<int> gt_considered;
  Mat_<unsigned short>::const_iterator pG = gt.begin<unsigned short>();
  Mat_<uchar>::const_iterator pL = label.begin<uchar>();
  Mat_<uchar>::const_iterator pI = inst.begin<uchar>();
  while(pG != gt.end<unsigned short>()) {
    int pGNum = static_cast<int>(*pG);
    int pINum = static_cast<int>(*pI);
    int pLNum = static_cast<int>(*pL);
    //cout << pGNum << "," << pINum << endl;
    // For every different gt instance, we need to calculate the intersection and union for every instance
    // We can make this faster by only considering when the intersections are nonzero
    if(pGNum != 0 && pINum != 0) {
      // Let's make sure this isn't an instance, gt pair we have already considered
      bool considered = false;
      std::pair<int, int> tmp(pGNum, pINum);
      if(std::find(gt_inst_considered.begin(), gt_inst_considered.end(), tmp) == gt_inst_considered.end()) {
        // We haven't considered this pair yet
        gt_inst_considered.push_back(tmp);
        float iou = CalcIoUPerInst(inst, gt, pINum, pGNum);
        //cout << "iou is: " << iou << endl;
        if (iou >= iou_thresh) {
          // Incrememnt positive count for label
          //cout << "Incrementing positive label " << pLNum << endl;
          (*positiveCountForLabel)[pLNum]++;
        }

      }
    }
    if(std::find(gt_considered.begin(), gt_considered.end(), pGNum) == gt_considered.end()) {
      gt_considered.push_back(pGNum);
      //cout << "Incrementing total label " << pLNum << endl;
      (*totalCountForLabel)[pLNum]++;
    }

    ++pG; ++pI; ++pL;
  }
}

int main(int argc, char* argv[]) {
  std::vector<int> positiveCountForLabel(256), totalCountForLabel(256);

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
    //cout << "inst type: " << type2str(inst.type());
    if (inst.empty()) {
      cout << "Error, couldn't load inst: " << inst_name << endl;
      break;
    }
    gt_name = regex_replace(line, regex("_leftImg8bit"), "_gtFine_instanceIds");
    gt_name = regex_replace(gt_name, regex("leftImg8bit"), "gtFine");
    gt = imread(gt_name, CV_LOAD_IMAGE_ANYDEPTH);
    //cout << "gt type: " << type2str(gt.type());
    if (gt.empty()) {
      cout << "Error, couldn't load gt: " << gt_name << endl;
      break;
    }
    label_name = regex_replace(line, regex("_leftImg8bit"), "_gtFine_labelIds");
    label_name = regex_replace(label_name, regex("leftImg8bit"), "gtFine");
    label = imread(label_name, CV_LOAD_IMAGE_GRAYSCALE);
    //cout << "label type: " << type2str(label.type());
    if (label.empty()) {
      cout << "Error, couldn't load label: " << label_name << endl;
      break;
    }
    
    //Let's calculate the intersection and union for each segment
    CalcIoU(inst, gt, label, FLAGS_iou_thresh, &positiveCountForLabel, &totalCountForLabel);

  }
  // Let's print out the results
  for(int i = 0; i < 256; i++) {
    int pC = positiveCountForLabel[i];
    int tC = totalCountForLabel[i];
    if(tC > 0) {
      float recall = static_cast<float>(pC) / static_cast<float>(tC);
      cout << "Label: " << i << ", Positives: " << pC << ", Total: " << tC << ", Recall: " << recall << endl;
    }
  }
  return 0;
}
