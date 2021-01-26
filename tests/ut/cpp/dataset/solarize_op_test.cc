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
#include "common/common.h"
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/solarize_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/util/status.h"
#include "utils/log_adapter.h"
#include "gtest/gtest.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestSolarizeOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestSolarizeOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

TEST_F(MindDataTestSolarizeOp, TestOp1) {
  MS_LOG(INFO) << "Doing testSolarizeOp1.";

  std::unique_ptr<SolarizeOp> op(new SolarizeOp());
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor_);

  EXPECT_TRUE(s.IsOk());
}

TEST_F(MindDataTestSolarizeOp, TestOp2) {
  MS_LOG(INFO) << "Doing testSolarizeOp2 - test default values";

  //  unsigned int threshold = 128;
  std::unique_ptr<SolarizeOp> op(new SolarizeOp());

  std::vector<uint8_t> test_vector = {3, 4, 59, 210, 255};
  std::vector<uint8_t> expected_output_vector = {252, 251, 196, 45, 0};
  std::shared_ptr<Tensor> test_input_tensor;
  std::shared_ptr<Tensor> expected_output_tensor;
  Tensor::CreateFromVector(test_vector, TensorShape({1, (long int)test_vector.size(), 1}), &test_input_tensor);
  Tensor::CreateFromVector(expected_output_vector, TensorShape({1, (long int)test_vector.size(), 1}),
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

TEST_F(MindDataTestSolarizeOp, TestOp3) {
  MS_LOG(INFO) << "Doing testSolarizeOp3 - Pass in only threshold_min parameter";

  //  unsigned int threshold = 128;
  std::vector<uint8_t> threshold ={1, 255};
  std::unique_ptr<SolarizeOp> op(new SolarizeOp(threshold));

  std::vector<uint8_t> test_vector = {3, 4, 59, 210, 255};
  std::vector<uint8_t> expected_output_vector = {252, 251, 196, 45, 0};
  std::shared_ptr<Tensor> test_input_tensor;
  std::shared_ptr<Tensor> expected_output_tensor;
  Tensor::CreateFromVector(test_vector, TensorShape({1, (long int)test_vector.size(), 1}), &test_input_tensor);
  Tensor::CreateFromVector(expected_output_vector, TensorShape({1, (long int)test_vector.size(), 1}),
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

TEST_F(MindDataTestSolarizeOp, TestOp4) {
  MS_LOG(INFO) << "Doing testSolarizeOp4 - Pass in both threshold parameters.";

  std::vector<uint8_t> threshold ={1, 230};
  std::unique_ptr<SolarizeOp> op(new SolarizeOp(threshold));

  std::vector<uint8_t> test_vector = {3, 4, 59, 210, 255};
  std::vector<uint8_t> expected_output_vector = {252, 251, 196, 45, 255};
  std::shared_ptr<Tensor> test_input_tensor;
  std::shared_ptr<Tensor> expected_output_tensor;
  Tensor::CreateFromVector(test_vector, TensorShape({1, (long int)test_vector.size(), 1}), &test_input_tensor);
  Tensor::CreateFromVector(expected_output_vector, TensorShape({1, (long int)test_vector.size(), 1}),
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

TEST_F(MindDataTestSolarizeOp, TestOp5) {
  MS_LOG(INFO) << "Doing testSolarizeOp5 - Rank 2 input tensor.";

  std::vector<uint8_t> threshold ={1, 230};
  std::unique_ptr<SolarizeOp> op(new SolarizeOp(threshold));

  std::vector<uint8_t> test_vector = {3, 4, 59, 210, 255};
  std::vector<uint8_t> expected_output_vector = {252, 251, 196, 45, 255};
  std::shared_ptr<Tensor> test_input_tensor;
  std::shared_ptr<Tensor> expected_output_tensor;
  Tensor::CreateFromVector(test_vector, TensorShape({1, (long int)test_vector.size()}), &test_input_tensor);
  Tensor::CreateFromVector(expected_output_vector, TensorShape({1, (long int)test_vector.size()}),
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

TEST_F(MindDataTestSolarizeOp, TestOp6) {
  MS_LOG(INFO) << "Doing testSolarizeOp6 - Bad Input.";

  std::vector<uint8_t> threshold ={10, 1};
  std::unique_ptr<SolarizeOp> op(new SolarizeOp(threshold));

  std::vector<uint8_t> test_vector = {3, 4, 59, 210, 255};
  std::shared_ptr<Tensor> test_input_tensor;
  std::shared_ptr<Tensor> test_output_tensor;
  Tensor::CreateFromVector(test_vector, TensorShape({1, (long int)test_vector.size(), 1}), &test_input_tensor);

  Status s = op->Compute(test_input_tensor, &test_output_tensor);

  EXPECT_TRUE(s.IsError());
  EXPECT_NE(s.ToString().find("Solarize: threshold_min must be smaller or equal to threshold_max."),
          std::string::npos);
  ASSERT_TRUE(s.StatusCode() == StatusCode::kMDUnexpectedError);
}