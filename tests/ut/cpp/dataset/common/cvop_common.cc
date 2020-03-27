/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <memory>
#include <string>
#include <vector>
#include "cvop_common.h"
#include "dataset/core/constants.h"
#include "common/utils.h"
#include "dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"
#include <fstream>
#include <opencv2/opencv.hpp>

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;
using UT::CVOP::CVOpCommon;

CVOpCommon::CVOpCommon() {}

CVOpCommon::~CVOpCommon() {}

void CVOpCommon::SetUp() {
  MS_LOG(INFO) << "starting test.";
  filename_ = GetFilename();
  GetInputImage(filename_);
}

std::string CVOpCommon::GetFilename() {
  std::string de_home = "data/dataset";
  std::string filename = de_home + "/apple.jpg";
  MS_LOG(INFO) << "Reading " << common::SafeCStr(filename) << ".";
  return filename;
}

void CVOpCommon::GetInputImage(std::string filename) {
  try {
    std::ifstream tmp(filename, std::ios::binary | std::ios::ate);
    dsize_t file_size = tmp.tellg();
    tmp.close();

    std::ifstream file(filename, std::ios::binary);
    TensorShape in_shape({file_size});
    raw_input_tensor_ = std::make_shared<Tensor>(in_shape, DataType(DataType::DE_UINT8));

    file.read(reinterpret_cast<char *>(raw_input_tensor_->StartAddr()), raw_input_tensor_->SizeInBytes());
    raw_cv_image_ = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
    input_tensor_ = std::dynamic_pointer_cast<Tensor>(std::make_shared<CVTensor>(raw_cv_image_));
    SwapRedAndBlue(input_tensor_, &input_tensor_);
    if (raw_cv_image_.data) {
      MS_LOG(INFO) << "Reading was successful. Height:" << raw_cv_image_.rows << " Width: " << raw_cv_image_.cols
                   << " Channels:" << raw_cv_image_.channels() << ".";
    }
  } catch (...) {
    MS_LOG(INFO) << "Error in GetInputImage.";
  }
}

void CVOpCommon::Save(const std::shared_ptr<Tensor> &tensor, std::string filename) {
  std::shared_ptr<Tensor> output;
  SwapRedAndBlue(tensor, &output);

  cv::Mat output_image_CV = std::dynamic_pointer_cast<CVTensor>(output)->mat();
  cv::imwrite(filename, output_image_CV);
}

std::string CVOpCommon::GetJPGStr(const cv::Mat &image) {
  std::vector<unsigned char> buff_jpg;
  cv::imencode(".jpg", image, buff_jpg);
  return std::string(buff_jpg.begin(), buff_jpg.end());
}

bool CVOpCommon::CompareCVMat(const cv::Mat &actual, const cv::Mat &expect, OperatorType type) {
  if (actual.size() != expect.size() || actual.channels() != expect.channels() || actual.type() != expect.type()) {
    return false;
  }
  std::string strJPGData_actual = GetJPGStr(actual);
  std::string strJPGData_expect = GetJPGStr(expect);

  bool success = strJPGData_actual.compare(strJPGData_expect) == 0;
  if (type == kFlipHorizontal || type == kFlipVertical) {
    std::string raw_filename = filename_;
    raw_filename.replace(raw_filename.end() - 9, raw_filename.end(), "imagefolder/apple_expect_not_flip.jpg");
    cv::Mat raw;
    raw = cv::imread(raw_filename, cv::ImreadModes::IMREAD_COLOR);
    std::string strJPGData_raw = GetJPGStr(raw);
    success = success || strJPGData_actual.compare(strJPGData_raw) == 0;
  } else if (type == kCrop) {
    success = expect.rows == actual.rows && expect.cols == actual.cols && expect.channels(), actual.channels();
  }
  return success;
}

void CVOpCommon::CheckImageShapeAndData(const std::shared_ptr<Tensor> &output_tensor, OperatorType type) {
  std::string dir_path = filename_.substr(0, filename_.length() - 9);
  std::string expect_image_path, actual_image_path;
  switch (type) {
    case kRescale:
      expect_image_path = dir_path + "imagefolder/apple_expect_rescaled.jpg";
      actual_image_path = dir_path + "imagefolder/apple_actual_rescaled.jpg";
      break;
    case kResizeBilinear:
      expect_image_path = dir_path + "imagefolder/apple_expect_resize_bilinear.jpg";
      actual_image_path = dir_path + "imagefolder/apple_actual_resize_bilinear.jpg";
      break;
    case kFlipHorizontal:
      expect_image_path = dir_path + "imagefolder/apple_expect_flipped_horizontal.jpg";
      actual_image_path = dir_path + "imagefolder/apple_actual_flipped_horizontal.jpg";
      break;
    case kFlipVertical:
      expect_image_path = dir_path + "imagefolder/apple_expect_flipped_vertical.jpg";
      actual_image_path = dir_path + "imagefolder/apple_actual_flipped_vertical.jpg";
      break;
    case kDecode:
      expect_image_path = dir_path + "imagefolder/apple_expect_decoded.jpg";
      actual_image_path = dir_path + "imagefolder/apple_actual_decoded.jpg";
      break;
    case kChangeMode:
      expect_image_path = dir_path + "imagefolder/apple_expect_changemode.jpg";
      actual_image_path = dir_path + "imagefolder/apple_actual_changemode.jpg";
      break;
    default:
      MS_LOG(INFO) << "Not pass verification! Operation type does not exists.";
      EXPECT_EQ(0, 1);
  }
  cv::Mat expect_img = cv::imread(expect_image_path, cv::IMREAD_COLOR);
  cv::Mat actual_img;
  // Saving
  MS_LOG(INFO) << "output_tensor.shape is " << output_tensor->shape();
  Save(output_tensor, actual_image_path);
  actual_img = cv::imread(actual_image_path, cv::IMREAD_COLOR);
  if (actual_img.data) {
    EXPECT_EQ(CompareCVMat(actual_img, expect_img, type), true);
  } else {
    MS_LOG(INFO) << "Not pass verification! Image data is null.";
    EXPECT_EQ(0, 1);
  }
}
