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
#include "common/common.h"
#include "minddata/dataset/kernels/data/fill_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestFillOp : public UT::Common {
 protected:
  MindDataTestFillOp() {}
};

TEST_F(MindDataTestFillOp, TestOp) {
  MS_LOG(INFO) << "Doing MindDataTestFillOp-TestOp.";
  std::vector<uint64_t> labels = {1, 1, 2};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, &input);

  std::shared_ptr<Tensor> fill_tensor;
  Tensor::CreateScalar<uint64_t>(4, &fill_tensor);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {4, 4, 4};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());
  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestFillOp-TestOp end.";
}

TEST_F(MindDataTestFillOp, TestCasting) {
  MS_LOG(INFO) << "Doing MindDataTestFillOp-TestCasting.";
  std::vector<uint64_t> labels = {0, 1, 2};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, &input);

  std::shared_ptr<Tensor> fill_tensor;
  Tensor::CreateScalar<float>(2.0, &fill_tensor);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {2, 2, 2};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, &expected);

  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  EXPECT_TRUE(s.IsOk());
  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;
  ASSERT_TRUE(*output == *expected);

  MS_LOG(INFO) << "MindDataTestFillOp-TestCasting end.";
}

TEST_F(MindDataTestFillOp, ScalarFill) {
  MS_LOG(INFO) << "Doing MindDataTestFillOp-ScalarFill.";
  std::vector<uint64_t> labels = {0, 1, 2};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, &input);

  TensorShape fill_shape({2});
  std::vector<uint64_t> fill_labels = {0, 1};
  std::shared_ptr<Tensor> fill_tensor;
  Tensor::CreateFromVector(fill_labels, &fill_tensor);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  EXPECT_TRUE(s.IsError());
  ASSERT_TRUE(s.StatusCode() == StatusCode::kMDUnexpectedError);

  MS_LOG(INFO) << "MindDataTestFillOp-ScalarFill end.";
}

TEST_F(MindDataTestFillOp, StringFill) {
  MS_LOG(INFO) << "Doing MindDataTestFillOp-StringFill.";
  std::vector<std::string> strings = {"xyzzy", "plugh", "abracadabra"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(strings, &input);

  std::shared_ptr<Tensor> fill_tensor;
  Tensor::CreateScalar<std::string>("hello", &fill_tensor);

  std::shared_ptr<Tensor> output;

  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  std::vector<std::string> expected_strings = {"hello", "hello", "hello"};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(expected_strings, &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());
  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);

  MS_LOG(INFO) << "MindDataTestFillOp-StringFill end.";
}

TEST_F(MindDataTestFillOp, NumericToString) {
  MS_LOG(INFO) << "Doing MindDataTestFillOp-NumericToString.";
  std::vector<std::string> strings = {"xyzzy", "plugh", "abracadabra"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(strings, &input);

  std::shared_ptr<Tensor> fill_tensor;
  Tensor::CreateScalar<float>(2.0, &fill_tensor);

  std::shared_ptr<Tensor> output;

  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  EXPECT_TRUE(s.IsError());
  ASSERT_TRUE(s.StatusCode() == StatusCode::kMDUnexpectedError);

  MS_LOG(INFO) << "MindDataTestFillOp-NumericToString end.";
}

TEST_F(MindDataTestFillOp, StringToNumeric) {
  MS_LOG(INFO) << "Doing MindDataTestFillOp-StringToNumeric.";
  std::vector<uint64_t> labels = {0, 1, 2};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, &input);

  std::shared_ptr<Tensor> fill_tensor;
  Tensor::CreateScalar<std::string>("hello", &fill_tensor);

  std::shared_ptr<Tensor> output;

  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  EXPECT_TRUE(s.IsError());
  ASSERT_TRUE(s.StatusCode() == StatusCode::kMDUnexpectedError);

  MS_LOG(INFO) << "MindDataTestFillOp-StringToNumeric end.";
}