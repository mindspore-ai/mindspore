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
#include "minddata/dataset/kernels/image/random_rotation_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestRandomRotationOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomRotationOp() : CVOpCommon() {}
};

/// Feature: RandomRotation op
/// Description: Test RandomRotationOp with basic usage using empty vector for center and check OneToOne
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestRandomRotationOp, TestOp) {
  MS_LOG(INFO) << "Doing MindDataTestRandomRotationOp::TestOp.";
  std::shared_ptr<Tensor> output_tensor;
  float sDegree = -180;
  float eDegree = 180;
  // use compute center to use for rotation
  std::vector<float> center = {};
  bool expand = false;
  auto op = std::make_unique<RandomRotationOp>(sDegree, eDegree, InterpolationMode::kLinear, expand, center, 0, 0, 0);
  EXPECT_TRUE(op->OneToOne());
  auto input_shape = input_tensor_->shape();
  EXPECT_EQ(op->Compute(input_tensor_, &output_tensor), Status::OK());
  EXPECT_EQ(input_shape.Size(), 3);
  EXPECT_EQ(output_tensor->shape().Size(), 3);
  EXPECT_EQ(input_shape[0], output_tensor->shape()[0]);
  EXPECT_EQ(input_shape[1], output_tensor->shape()[1]);
}
