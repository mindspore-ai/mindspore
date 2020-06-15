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
#include "dataset/kernels/data/fill_op.h"
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
  uint64_t labels[3] = {1, 1, 2};
  TensorShape shape({3});
  std::shared_ptr<Tensor> input =
    std::make_shared<Tensor>(shape, DataType(DataType::DE_UINT64), reinterpret_cast<unsigned char *>(labels));

  TensorShape fill_shape({});
  std::shared_ptr<Tensor> fill_tensor = std::make_shared<Tensor>(fill_shape, DataType(DataType::DE_UINT64));
  fill_tensor->SetItemAt<uint64_t>({}, 4);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  uint64_t out[3] = {4, 4, 4};

  std::shared_ptr<Tensor> expected =
    std::make_shared<Tensor>(TensorShape{3}, DataType(DataType::DE_UINT64), reinterpret_cast<unsigned char *>(out));

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
  uint64_t labels[3] = {0, 1, 2};
  TensorShape shape({3});
  std::shared_ptr<Tensor> input =
    std::make_shared<Tensor>(shape, DataType(DataType::DE_UINT64), reinterpret_cast<unsigned char *>(labels));

  TensorShape fill_shape({});
  std::shared_ptr<Tensor> fill_tensor = std::make_shared<Tensor>(fill_shape, DataType(DataType::DE_FLOAT32));
  fill_tensor->SetItemAt<float>({}, 2.0);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  uint64_t out[3] = {2, 2, 2};

  std::shared_ptr<Tensor> expected =
    std::make_shared<Tensor>(TensorShape{3}, DataType(DataType::DE_UINT64), reinterpret_cast<unsigned char *>(out));

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
  uint64_t labels[3] = {0, 1, 2};
  TensorShape shape({3});
  std::shared_ptr<Tensor> input =
    std::make_shared<Tensor>(shape, DataType(DataType::DE_UINT64), reinterpret_cast<unsigned char *>(labels));

  TensorShape fill_shape({2});
  uint64_t fill_labels[3] = {0, 1};
  std::shared_ptr<Tensor> fill_tensor =
    std::make_shared<Tensor>(fill_shape, DataType(DataType::DE_UINT64), reinterpret_cast<unsigned char *>(fill_labels));
  std::shared_ptr<Tensor> output;
  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  EXPECT_TRUE(s.IsError());
  ASSERT_TRUE(s.get_code() == StatusCode::kUnexpectedError);

  MS_LOG(INFO) << "MindDataTestFillOp-ScalarFill end.";
}

TEST_F(MindDataTestFillOp, StringFill) {
  MS_LOG(INFO) << "Doing MindDataTestFillOp-StringFill.";
  std::vector<std::string> strings = {"xyzzy", "plugh", "abracadabra"};
  TensorShape shape({3});
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>(strings, shape);

  TensorShape fill_shape({});
  std::string fill_string = "hello";
  std::shared_ptr<Tensor> fill_tensor = std::make_shared<Tensor>(fill_string);

  std::shared_ptr<Tensor> output;

  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  std::vector<std::string> expected_strings = {"hello", "hello", "hello"};
  TensorShape expected_shape({3});
  std::shared_ptr<Tensor> expected = std::make_shared<Tensor>(expected_strings, expected_shape);

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
  TensorShape shape({3});
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>(strings, shape);

  TensorShape fill_shape({});
  std::shared_ptr<Tensor> fill_tensor = std::make_shared<Tensor>(fill_shape, DataType(DataType::DE_FLOAT32));
  fill_tensor->SetItemAt<float>({}, 2.0);

  std::shared_ptr<Tensor> output;

  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  EXPECT_TRUE(s.IsError());
  ASSERT_TRUE(s.get_code() == StatusCode::kUnexpectedError);

  MS_LOG(INFO) << "MindDataTestFillOp-NumericToString end.";
}

TEST_F(MindDataTestFillOp, StringToNumeric) {
  MS_LOG(INFO) << "Doing MindDataTestFillOp-StringToNumeric.";
  uint64_t labels[3] = {0, 1, 2};
  TensorShape shape({3});
  std::shared_ptr<Tensor> input =
    std::make_shared<Tensor>(shape, DataType(DataType::DE_UINT64), reinterpret_cast<unsigned char *>(labels));

  TensorShape fill_shape({});
  std::string fill_string = "hello";
  std::shared_ptr<Tensor> fill_tensor = std::make_shared<Tensor>(fill_string);

  std::shared_ptr<Tensor> output;

  std::unique_ptr<FillOp> op(new FillOp(fill_tensor));
  Status s = op->Compute(input, &output);

  EXPECT_TRUE(s.IsError());
  ASSERT_TRUE(s.get_code() == StatusCode::kUnexpectedError);

  MS_LOG(INFO) << "MindDataTestFillOp-StringToNumeric end.";
}