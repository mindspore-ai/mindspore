/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bboxop_common.h"

#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include <stdio.h>

#include "./tinyxml2.h"
#include "opencv2/opencv.hpp"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using namespace UT::CVOP::BBOXOP;
using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLError;

const char kAnnotationsFolder[] = "/Annotations/";
const char kImagesFolder[] = "/JPEGImages/";
const char kExpectedName[] = "apple_expect_";
const char kActualName[] = "Actual";
const char kAnnotExt[] = ".xml";
const char kImageExt[] = ".jpg";

BBoxOpCommon::BBoxOpCommon() {}

BBoxOpCommon::~BBoxOpCommon() {}

void BBoxOpCommon::SetUp() {
  MS_LOG(INFO) << "starting test.";
  image_folder_build_ = "data/dataset/imagefolder/";
  image_folder_src_ = "../../../../../tests/ut/data/dataset/imagefolder/";
  std::string dir_path = "data/dataset/testVOC2012_2";
  GetInputImagesAndAnnotations(dir_path);
}

void BBoxOpCommon::GetInputImagesAndAnnotations(const std::string &dir, std::size_t num_of_samples) {
  std::string images_path = dir + std::string(kImagesFolder);
  std::string annots_path = dir + std::string(kAnnotationsFolder);
  Path dir_path(images_path);
  std::shared_ptr<Path::DirIterator> image_dir_itr = Path::DirIterator::OpenDirectory(&dir_path);
  std::vector<std::string> paths_to_fetch;
  if (!dir_path.Exists()) {
    MS_LOG(ERROR) << "Images folder was not found : " + images_path;
    EXPECT_TRUE(dir_path.Exists());
  }
  // get image file paths
  while (image_dir_itr->HasNext()) {
    Path image_path = image_dir_itr->Next();
    if (image_path.Extension() == std::string(kImageExt)) {
      paths_to_fetch.push_back(image_path.ToString());
    }
  }
  // sort fetched files
  std::sort(paths_to_fetch.begin(), paths_to_fetch.end());
  std::size_t files_fetched = 0;
  for (const auto &image_file : paths_to_fetch) {
    std::string image_ext = std::string(kImageExt);
    std::string annot_file = image_file;
    std::size_t pos = 0;
    // first replace the Image dir with the Annotation dir.
    if ((pos = image_file.find(std::string(kImagesFolder), 0)) != std::string::npos) {
      annot_file.replace(pos, std::string(kImagesFolder).length(), std::string(kAnnotationsFolder));
    }
    // then replace the extensions. the image extension to annotation extension
    if ((pos = annot_file.find(image_ext, 0)) != std::string::npos) {
      annot_file.replace(pos, std::string(kAnnotExt).length(), std::string(kAnnotExt));
    }
    std::shared_ptr<Tensor> annotation_tensor;
    // load annotations and log failure
    if (!LoadAnnotationFile(annot_file, &annotation_tensor)) {
      MS_LOG(ERROR) << "Loading Annotations failed in GetInputImagesAndAnnotations";
      EXPECT_EQ(0, 1);
    }
    // load image
    GetInputImage(image_file);
    // add image and annotation to the tensor table
    std::shared_ptr<Tensor> input1;
    Tensor::CreateFromTensor(input_tensor_, &input1);
    TensorRow row_data({std::move(input1), std::move(annotation_tensor)});
    images_and_annotations_.push_back(row_data);
    files_fetched++;
    if (files_fetched == num_of_samples) {
      break;
    }
  }
}

void BBoxOpCommon::SaveImagesWithAnnotations(BBoxOpCommon::FileType type, const std::string &op_name,
                                             const TensorTable &table) {
  int i = 0;
  for (auto &row : table) {
    std::shared_ptr<Tensor> row_to_save;
    // fix: data race by SwapRedAndBlue
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(row[0]);
    cv::Mat rgb = BGRToRGB(input_cv->mat());
    cv::Mat image;
    rgb.copyTo(image);
    uint32_t num_of_boxes = row[1]->shape()[0];
    bool passing_data_fetch = true;
    // For each bounding box draw on the image.
    for (uint32_t i = 0; i < num_of_boxes; i++) {
      float x = 0.0, y = 0.0, w = 0.0, h = 0.0;
      passing_data_fetch &= row[1]->GetItemAt<float>(&x, {i, 0}).IsOk();
      passing_data_fetch &= row[1]->GetItemAt<float>(&y, {i, 1}).IsOk();
      passing_data_fetch &= row[1]->GetItemAt<float>(&w, {i, 2}).IsOk();
      passing_data_fetch &= row[1]->GetItemAt<float>(&h, {i, 3}).IsOk();
      if (!passing_data_fetch) {
        MS_LOG(ERROR) << "Fetching bbox coordinates failed in SaveImagesWithAnnotations.";
        EXPECT_TRUE(passing_data_fetch);
      }
      cv::Rect bbox(x, y, w, h);
      cv::rectangle(image, bbox, cv::Scalar(255, 0, 0), 10, 8, 0);
    }
    bool im_write_success = false;
    // if user wants to save an expected image, use the path to the source folder.
    if (type == FileType::kExpected) {
      im_write_success = cv::imwrite(
        image_folder_src_ + std::string(kExpectedName) + op_name + std::to_string(i) + std::string(kImageExt), image);
    } else {
      // otherwise if we are saving actual images only for comparison, save in current working dir in build folders.
      im_write_success =
        cv::imwrite(std::string(kActualName) + op_name + std::to_string(i) + std::string(kImageExt), image);
    }
    if (!im_write_success) {
      MS_LOG(ERROR) << "Image write failed in SaveImagesWithAnnotations.";
      EXPECT_TRUE(im_write_success);
    }
    i += 1;
  }
}

void BBoxOpCommon::CompareActualAndExpected(const std::string &op_name) {
  size_t num_of_images = images_and_annotations_.size();
  for (size_t i = 0; i < num_of_images; i++) {
    // load actual and expected images.
    std::string actual_path = std::string(kActualName) + op_name + std::to_string(i) + std::string(kImageExt);
    std::string expected_path =
      image_folder_build_ + std::string(kExpectedName) + op_name + std::to_string(i) + std::string(kImageExt);
    cv::Mat expect_img = cv::imread(expected_path, cv::IMREAD_COLOR);
    cv::Mat actual_img = cv::imread(actual_path, cv::IMREAD_COLOR);
    // after comparison is done remove temporary file
    EXPECT_TRUE(remove(actual_path.c_str()) == 0);
    // compare using ==operator by Tensor
    std::shared_ptr<CVTensor> expect_img_t, actual_img_t;
    CVTensor::CreateFromMat(expect_img, 3, &expect_img_t);
    CVTensor::CreateFromMat(actual_img, 3, &actual_img_t);
    if (actual_img.data) {
      EXPECT_EQ(*expect_img_t == *actual_img_t, true);
    } else {
      MS_LOG(ERROR) << "Not pass verification! Image data is null.";
      EXPECT_EQ(0, 1);
    }
  }
}

bool BBoxOpCommon::LoadAnnotationFile(const std::string &path, std::shared_ptr<Tensor> *target_BBox) {
  if (!Path(path).Exists()) {
    MS_LOG(ERROR) << "File is not found : " + path;
    return false;
  }
  XMLDocument doc;
  XMLError e = doc.LoadFile(mindspore::common::SafeCStr(path));
  if (e != XMLError::XML_SUCCESS) {
    MS_LOG(ERROR) << "Xml load failed";
    return false;
  }
  XMLElement *root = doc.RootElement();
  if (root == nullptr) {
    MS_LOG(ERROR) << "Xml load root element error";
    return false;
  }
  XMLElement *object = root->FirstChildElement("object");
  if (object == nullptr) {
    MS_LOG(ERROR) << "No object find in " + path;
    return false;
  }
  std::vector<float> return_value_list;
  dsize_t bbox_count = 0;      // keep track of number of bboxes in file
  dsize_t bbox_val_count = 4;  // creating bboxes of size 4 to test function
  // FILE OK TO READ
  while (object != nullptr) {
    bbox_count += 1;
    std::string label_name;
    float xmin = 0.0, ymin = 0.0, xmax = 0.0, ymax = 0.0;
    XMLElement *bbox_node = object->FirstChildElement("bndbox");
    if (bbox_node != nullptr) {
      XMLElement *xmin_node = bbox_node->FirstChildElement("xmin");
      if (xmin_node != nullptr) xmin = xmin_node->FloatText();
      XMLElement *ymin_node = bbox_node->FirstChildElement("ymin");
      if (ymin_node != nullptr) ymin = ymin_node->FloatText();
      XMLElement *xmax_node = bbox_node->FirstChildElement("xmax");
      if (xmax_node != nullptr) xmax = xmax_node->FloatText();
      XMLElement *ymax_node = bbox_node->FirstChildElement("ymax");
      if (ymax_node != nullptr) ymax = ymax_node->FloatText();
    } else {
      MS_LOG(ERROR) << "bndbox dismatch in " + path;
      return false;
    }
    if (xmin > 0 && ymin > 0 && xmax > xmin && ymax > ymin) {
      for (auto item : {xmin, ymin, xmax - xmin, ymax - ymin}) {
        return_value_list.push_back(item);
      }
    }
    object = object->NextSiblingElement("object");  // Read next BBox if exists
  }
  std::shared_ptr<Tensor> ret_value;
  Status s = Tensor::CreateFromVector(return_value_list, TensorShape({bbox_count, bbox_val_count}), &ret_value);
  EXPECT_TRUE(s.IsOk());
  (*target_BBox) = ret_value;  // load bbox from file into return
  return true;
}
