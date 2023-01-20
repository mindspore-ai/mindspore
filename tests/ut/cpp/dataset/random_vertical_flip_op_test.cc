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
#include "common/common.h"
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/random_vertical_flip_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestRandomVerticalFlipOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestRandomVerticalFlipOp() : CVOpCommon() {}
};

/// Feature: RandomVerticalFlip op
/// Description: Test RandomVerticalFlipOp with prob=0.5
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestRandomVerticalFlipOp, TestOp) {
  MS_LOG(INFO) << "Doing testVerticalFlip.";
  // flip
  TensorRow input_tensor_row;
  std::shared_ptr<Tensor> input1;
  Tensor::CreateFromTensor(input_tensor_, &input1);
  input_tensor_row.push_back(input1);
  std::shared_ptr<Tensor> input2;
  Tensor::CreateFromTensor(input_tensor_, &input2);
  input_tensor_row.push_back(input2);
  TensorRow output_tensor_row;
  auto op = std::make_unique<RandomVerticalFlipOp>(0.5);
  Status s = op->Compute(input_tensor_row, &output_tensor_row);
  EXPECT_TRUE(s.IsOk());
  CheckImageShapeAndData(input_tensor_, kFlipVertical);
  MS_LOG(INFO) << "testVerticalFlip end.";
}
