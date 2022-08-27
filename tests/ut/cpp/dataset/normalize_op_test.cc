/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "common/common.h"
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/image/normalize_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"
#include <opencv2/opencv.hpp>

using namespace mindspore::dataset;

class MindDataTestNormalizeOP : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestNormalizeOP() : CVOpCommon() {}
};

/// Feature: Normalize
/// Description: Normalize the image and save
/// Expectation: normalized image saves successfully
TEST_F(MindDataTestNormalizeOP, TestOp) {
  MS_LOG(INFO) << "Doing TestNormalizeOp::TestOp2.";
  std::shared_ptr<Tensor> output_tensor;

  // Numbers are from the resnet50 model implementation
  std::vector<float> mean = {121.0, 115.0, 100.0};
  std::vector<float> std = {70.0, 68.0, 71.0};

  // Normalize Op
  std::unique_ptr<NormalizeOp> op = std::make_unique<NormalizeOp>(mean, std, true);
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());

  std::string output_filename = GetFilename();
  output_filename.replace(output_filename.end() - 8, output_filename.end(), "imagefolder/normalizeOpOut.yml");

  std::shared_ptr<CVTensor> p = CVTensor::AsCVTensor(output_tensor);
  cv::Mat cv_output_image;
  cv_output_image = p->mat();

  MS_LOG(DEBUG) << "Storing output file to : " << output_filename << std::endl;
  cv::FileStorage file(output_filename, cv::FileStorage::WRITE);
  file << "imageData" << cv_output_image;
}

/// Feature: Normalize
/// Description: Test Normalize with 4 dimension tensor
/// Expectation: The result is as expected
TEST_F(MindDataTestNormalizeOP, TestOp4Dim) {
  MS_LOG(INFO) << "Doing TestNormalizeOp-TestOp4Dim.";
  std::shared_ptr<Tensor> output_tensor;

  // construct a fake 4 dimension data
  std::shared_ptr<Tensor> input_tensor_cp;
  ASSERT_OK(Tensor::CreateFromTensor(input_tensor_, &input_tensor_cp));
  std::vector<std::shared_ptr<Tensor>> tensor_list;
  tensor_list.push_back(input_tensor_cp);
  tensor_list.push_back(input_tensor_cp);
  TensorShape shape = input_tensor_cp->shape();
  std::shared_ptr<Tensor> input_4d;
  ASSERT_OK(TensorVectorToBatchTensor(tensor_list, &input_4d));
  std::vector<float> mean = {121.0, 115.0, 100.0};
  std::vector<float> std = {70.0, 68.0, 71.0};

  // Normalize Op
  std::unique_ptr<NormalizeOp> op = std::make_unique<NormalizeOp>(mean, std, true);
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_4d, &output_tensor);
  EXPECT_TRUE(s.IsOk());

  std::string output_filename = GetFilename();
  output_filename.replace(output_filename.end() - 8, output_filename.end(), "imagefolder/normalizeOpVideoOut.yml");

  std::shared_ptr<CVTensor> p = CVTensor::AsCVTensor(output_tensor);
  cv::Mat cv_output_video;
  cv_output_video = p->mat();

  MS_LOG(DEBUG) << "Storing output file to : " << output_filename << std::endl;
  cv::FileStorage file(output_filename, cv::FileStorage::WRITE);
  file << "videoData" << cv_output_video;
}
