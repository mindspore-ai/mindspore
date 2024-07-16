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
#include "minddata/dataset/kernels/image/swap_red_blue_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestSwapRedBlueOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestSwapRedBlueOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

/// Feature: SwapRedBlue op
/// Description: Test SwapRedBlueOp with default parameters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestSwapRedBlueOp, TestOp1) {
  MS_LOG(INFO) << "Doing testSwapRedBlue.";
  // SwapRedBlue params
  auto op = std::make_unique<SwapRedBlueOp>();
  EXPECT_TRUE(op->OneToOne());
  auto input_shape = input_tensor_->shape();
  EXPECT_EQ(op->Compute(input_tensor_, &output_tensor_), Status::OK());
  EXPECT_EQ(input_shape.Size(), 3);
  EXPECT_EQ(output_tensor_->shape().Size(), 3);
  size_t input_item_size = input_shape[0] * input_shape[1] * input_shape[2];
  size_t output_item_size = output_tensor_->shape()[0] * output_tensor_->shape()[1] * output_tensor_->shape()[2];
  EXPECT_EQ(input_item_size, output_item_size);
}
