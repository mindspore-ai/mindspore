/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/text/kernels/sliding_window_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestSlidingWindowOp : public UT::Common {
 protected:
  MindDataTestSlidingWindowOp() {}
};

/// Feature: SlidingWindow op
/// Description: Test SlidingWindowOp's Compute
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestSlidingWindowOp, Compute) {
  MS_LOG(INFO) << "Doing MindDataTestSlidingWindowOp->Compute.";
  std::vector<std::string> strings = {"one", "two", "three", "four", "five", "six", "seven", "eight"};
  TensorShape shape({static_cast<dsize_t>(strings.size())});
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(strings, shape, &input);
  std::shared_ptr<Tensor> output;

  auto op = std::make_unique<SlidingWindowOp>(3, 0);
  Status s = op->Compute(input, &output);

  std::vector<std::string> out = {"one",  "two",  "three", "two",  "three", "four",  "three", "four",  "five",
                                  "four", "five", "six",   "five", "six",   "seven", "six",   "seven", "eight"};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({6, 3}), &expected);

  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());
  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;
  ASSERT_TRUE(*output == *expected);

  MS_LOG(INFO) << "MindDataTestSlidingWindowOp end.";
}

/// Feature: SlidingWindow op
/// Description: Test SlidingWindowOp's OutputShape
/// Expectation: Output's shape is equal to the expected output's shape
TEST_F(MindDataTestSlidingWindowOp, OutputShape) {
  MS_LOG(INFO) << "Doing MindDataTestSlidingWindowOp->OutputShape.";
  std::vector<std::string> strings = {"one", "two", "three", "four", "five", "six", "seven", "eight"};
  TensorShape shape({static_cast<dsize_t>(strings.size())});
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(strings, shape, &input);
  std::vector<TensorShape> input_shape = {input->shape()};
  std::vector<TensorShape> output_shape = {TensorShape({})};

  auto op = std::make_unique<SlidingWindowOp>(3, 0);
  Status s = op->OutputShape(input_shape, output_shape);

  MS_LOG(DEBUG) << "input_shape" << input_shape[0];
  MS_LOG(DEBUG) << "output_shape" << output_shape[0];
  ASSERT_TRUE(output_shape[0] == TensorShape({6, 3}));

  MS_LOG(INFO) << "MindDataTestSlidingWindowOp end.";
}
