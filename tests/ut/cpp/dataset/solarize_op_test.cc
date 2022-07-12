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
#include "common/cvop_common.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/solarize_op.h"
#include "minddata/dataset/util/status.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestSolarizeOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestSolarizeOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

/// Feature: SolarizeOp
/// Description: Test SolarizeOp basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestSolarizeOp, TestSolarizeOpBasicUsage) {
  MS_LOG(INFO) << "Doing MindDataTestSolarizeOp-TestSolarizeOpBasicUsage.";

  std::vector<float> threshold = {1, 255};
  auto op = std::make_unique<SolarizeOp>(threshold);

  std::vector<uint8_t> test_vector = {3, 4, 59, 210, 255};
  std::vector<uint8_t> expected_output_vector = {252, 251, 196, 45, 0};
  std::shared_ptr<Tensor> test_input_tensor;
  std::shared_ptr<Tensor> expected_output_tensor;
  Tensor::CreateFromVector(test_vector, TensorShape({1, static_cast<dsize_t>(test_vector.size()), 1}),
                           &test_input_tensor);
  Tensor::CreateFromVector(expected_output_vector, TensorShape({1, static_cast<dsize_t>(test_vector.size()), 1}),
                           &expected_output_tensor);

  std::shared_ptr<Tensor> test_output_tensor;
  Status s = op->Compute(test_input_tensor, &test_output_tensor);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(test_output_tensor->shape() == expected_output_tensor->shape());
  ASSERT_TRUE(test_output_tensor->type() == expected_output_tensor->type());
  MS_LOG(DEBUG) << *test_output_tensor << std::endl;
  MS_LOG(DEBUG) << *expected_output_tensor << std::endl;
  ASSERT_TRUE(*test_output_tensor == *expected_output_tensor);
}

/// Feature: SolarizeOp
/// Description: Test SolarizeOp with float type tensor
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestSolarizeOp, TestSolarizeOpFloat) {
  MS_LOG(INFO) << "Doing MindDataTestSolarizeOp-TestSolarizeOpFloat.";

  std::vector<float> threshold = {0.2, 0.8};
  auto op = std::make_unique<SolarizeOp>(threshold);

  std::vector<float> test_vector = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<float> expected_output_vector = {0.1, 254.7, 254.5, 254.3, 0.9};
  std::shared_ptr<Tensor> test_input_tensor;
  std::shared_ptr<Tensor> expected_output_tensor;
  Tensor::CreateFromVector(test_vector, TensorShape({1, static_cast<dsize_t>(test_vector.size()), 1}),
                           &test_input_tensor);
  Tensor::CreateFromVector(expected_output_vector, TensorShape({1, static_cast<dsize_t>(test_vector.size()), 1}),
                           &expected_output_tensor);

  std::shared_ptr<Tensor> test_output_tensor;
  Status s = op->Compute(test_input_tensor, &test_output_tensor);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(test_output_tensor->shape() == expected_output_tensor->shape());
  ASSERT_TRUE(test_output_tensor->type() == expected_output_tensor->type());
  MS_LOG(DEBUG) << *test_output_tensor << std::endl;
  MS_LOG(DEBUG) << *expected_output_tensor << std::endl;
  ASSERT_TRUE(*test_output_tensor == *expected_output_tensor);
}

/// Feature: SolarizeOp
/// Description: Test SolarizeOp on tensor with dim 2
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestSolarizeOp, TestSolarizeOpDim2) {
  MS_LOG(INFO) << "Doing MindDataTestSolarizeOp-TestSolarizeOpDim2.";

  std::vector<float> threshold = {1, 230};
  auto op = std::make_unique<SolarizeOp>(threshold);

  std::vector<uint8_t> test_vector = {3, 4, 59, 210, 255};
  std::vector<uint8_t> expected_output_vector = {252, 251, 196, 45, 255};
  std::shared_ptr<Tensor> test_input_tensor;
  std::shared_ptr<Tensor> expected_output_tensor;
  Tensor::CreateFromVector(test_vector, TensorShape({1, static_cast<dsize_t>(test_vector.size())}), &test_input_tensor);
  Tensor::CreateFromVector(expected_output_vector, TensorShape({1, static_cast<dsize_t>(test_vector.size())}),
                           &expected_output_tensor);

  std::shared_ptr<Tensor> test_output_tensor;
  Status s = op->Compute(test_input_tensor, &test_output_tensor);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(test_output_tensor->shape() == expected_output_tensor->shape());
  ASSERT_TRUE(test_output_tensor->type() == expected_output_tensor->type());
  MS_LOG(DEBUG) << *test_output_tensor << std::endl;
  MS_LOG(DEBUG) << *expected_output_tensor << std::endl;

  ASSERT_TRUE(*test_output_tensor == *expected_output_tensor);
}
