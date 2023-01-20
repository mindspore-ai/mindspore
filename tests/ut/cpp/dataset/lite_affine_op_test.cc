/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "common/cvop_common.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/affine_op.h"
#include "minddata/dataset/kernels/image/math_utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"

using namespace mindspore::dataset;
using mindspore::dataset::InterpolationMode;

class MindDataTestAffineOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestAffineOp() : CVOpCommon() {}
};

// Helper function, consider moving this to helper class for UT
double Mse(cv::Mat img1, cv::Mat img2) {
  // clone to get around open cv optimization
  cv::Mat output1 = img1.clone();
  cv::Mat output2 = img2.clone();

  // input check
  if (output1.rows < 0 || output1.rows != output2.rows || output1.cols < 0 || output1.cols != output2.cols) {
    return 10000.0;
  }
  return cv::norm(output1, output2, cv::NORM_L1);
}

// helper function to generate corresponding affine matrix
std::vector<double> GenerateMatrix(const std::shared_ptr<Tensor> &input, float_t degrees,
                                   const std::vector<float_t> &translation, float_t scale,
                                   const std::vector<float_t> &shear) {
  float_t translation_x = translation[0];
  float_t translation_y = translation[1];
  DegreesToRadians(degrees, &degrees);
  float_t shear_x = shear[0];
  float_t shear_y = shear[1];
  DegreesToRadians(shear_x, &shear_x);
  DegreesToRadians(-1 * shear_y, &shear_y);
  float_t cx = ((input->shape()[1] - 1) / 2.0);
  float_t cy = ((input->shape()[0] - 1) / 2.0);
  // Calculate RSS
  std::vector<double> matrix{
    static_cast<double>(scale * cos(degrees + shear_y) / cos(shear_y)),
    static_cast<double>(scale * (-1 * cos(degrees + shear_y) * tan(shear_x) / cos(shear_y) - sin(degrees))),
    0,
    static_cast<double>(scale * sin(degrees + shear_y) / cos(shear_y)),
    static_cast<double>(scale * (-1 * sin(degrees + shear_y) * tan(shear_x) / cos(shear_y) + cos(degrees))),
    0};
  // Compute T * C * RSS * C^-1
  matrix[2] = (1 - matrix[0]) * cx - matrix[1] * cy + translation_x;
  matrix[5] = (1 - matrix[4]) * cy - matrix[3] * cx + translation_y;
  return matrix;
}

/// Feature: Affine op
/// Description: Test Affine op using lite_mat_rgb
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestAffineOp, TestAffineLite) {
  MS_LOG(INFO) << "Doing MindDataTestAffine-TestAffineLite.";

  // create input tensor and
  float degree = 0.0;
  std::vector<float> translation = {0.0, 0.0};
  float scale = 0.0;
  std::vector<float> shear = {0.0, 0.0};

  // Create affine object with default values
  auto op = std::make_shared<AffineOp>(degree, translation, scale, shear, InterpolationMode::kLinear);
  // output tensor
  std::shared_ptr<Tensor> output_tensor;

  // output
  LiteMat dst;
  LiteMat lite_mat_rgb(input_tensor_->shape()[1], input_tensor_->shape()[0], input_tensor_->shape()[2],
                       const_cast<void *>(reinterpret_cast<const void *>(input_tensor_->GetBuffer())),
                       LDataType::UINT8);

  std::vector<double> matrix = GenerateMatrix(input_tensor_, degree, translation, scale, shear);

  int height = lite_mat_rgb.height_;
  int width = lite_mat_rgb.width_;
  std::vector<size_t> dsize;
  dsize.push_back(width);
  dsize.push_back(height);
  double M[6] = {};
  for (int i = 0; i < matrix.size(); i++) {
    M[i] = static_cast<double>(matrix[i]);
  }

  EXPECT_TRUE(Affine(lite_mat_rgb, dst, M, dsize, UINT8_C3(0, 0, 0)));
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  // output tensor is a cv tenosr, we can compare mat values
  cv::Mat lite_cv_out(dst.height_, dst.width_, CV_8UC3, dst.data_ptr_);
  double mse = Mse(lite_cv_out, CVTensor(output_tensor).mat());
  MS_LOG(INFO) << "mse: " << std::to_string(mse) << std::endl;
  EXPECT_LT(mse, 1);  // predetermined magic number
}
