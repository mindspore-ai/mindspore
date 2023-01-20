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
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/pad_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestPadOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestPadOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

/// Feature: Pad op
/// Description: Test PadOp basic usage and check OneToOne
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPadOp, TestOp) {
  MS_LOG(INFO) << "Doing testPad.";
  auto op = std::make_unique<PadOp>(10, 20, 30, 40, BorderType::kConstant);
  EXPECT_TRUE(op->OneToOne());
  std::shared_ptr<Tensor> input1;
  Tensor::CreateFromTensor(input_tensor_, &input1);
  Status s = op->Compute(input1, &output_tensor_);
  size_t actual = 0;
  if (s == Status::OK()) {
    actual = output_tensor_->shape()[0] * output_tensor_->shape()[1] * output_tensor_->shape()[2];
  }
  EXPECT_EQ(actual, (input_tensor_->shape()[0] + 30) * (input_tensor_->shape()[1] + 70) * 3);
  EXPECT_EQ(s, Status::OK());
}
