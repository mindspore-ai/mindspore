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
#include "minddata/dataset/kernels/data/concatenate_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestConcatenateOp : public UT::Common {
 protected:
  MindDataTestConcatenateOp() {}
};

TEST_F(MindDataTestConcatenateOp, TestOp) {
  MS_LOG(INFO) << "Doing MindDataTestConcatenate-TestOp-SingleRowinput.";
  std::vector<uint64_t> labels = {1, 1, 2};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, &input);

  std::vector<uint64_t> append_labels = {4, 4, 4};
  std::shared_ptr<Tensor> append;
  Tensor::CreateFromVector(append_labels, &append);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<ConcatenateOp> op(new ConcatenateOp(0, nullptr, append));
  TensorRow in;
  in.push_back(input);
  TensorRow out_row;
  Status s = op->Compute(in, &out_row);
  std::vector<uint64_t> out = {1, 1, 2, 4, 4, 4};

  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, &expected);

  output = out_row[0];
  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());
  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;
  ASSERT_TRUE(*output == *expected);
}

TEST_F(MindDataTestConcatenateOp, TestOp2) {
  MS_LOG(INFO) << "Doing MindDataTestConcatenate-TestOp2-MultiInput.";
  std::vector<uint64_t> labels = {1, 12, 2};
  std::shared_ptr<Tensor> row_1;
  Tensor::CreateFromVector(labels, &row_1);

  std::shared_ptr<Tensor> row_2;
  Tensor::CreateFromVector(labels, &row_2);

  std::vector<uint64_t> append_labels = {4, 4, 4};
  std::shared_ptr<Tensor> append;
  Tensor::CreateFromVector(append_labels, &append);

  TensorRow tensor_list;
  tensor_list.push_back(row_1);
  tensor_list.push_back(row_2);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<ConcatenateOp> op(new ConcatenateOp(0, nullptr, append));

  TensorRow out_row;
  Status s = op->Compute(tensor_list, &out_row);
  std::vector<uint64_t> out = {1, 12, 2, 1, 12, 2, 4, 4, 4};

  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, &expected);

  output = out_row[0];
  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());
  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;
  ASSERT_TRUE(*output == *expected);
}

TEST_F(MindDataTestConcatenateOp, TestOp3) {
  MS_LOG(INFO) << "Doing MindDataTestConcatenate-TestOp3-Strings.";
  std::vector<std::string> labels = {"hello", "bye"};
  std::shared_ptr<Tensor> row_1;
  Tensor::CreateFromVector(labels, &row_1);

  std::vector<std::string> append_labels = {"1", "2", "3"};
  std::shared_ptr<Tensor> append;
  Tensor::CreateFromVector(append_labels, &append);

  TensorRow tensor_list;
  tensor_list.push_back(row_1);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<ConcatenateOp> op(new ConcatenateOp(0, nullptr, append));

  TensorRow out_row;
  Status s = op->Compute(tensor_list, &out_row);
  std::vector<std::string> out = {"hello", "bye", "1", "2", "3"};

  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, &expected);

  output = out_row[0];
  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());
  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;
  ASSERT_TRUE(*output == *expected);
}
