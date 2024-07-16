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
#include "minddata/dataset/kernels/data/to_float16_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestToFloat16Op : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestToFloat16Op() : CVOpCommon() {}
};

/// Feature: ToFloat16 op
/// Description: Test ToFloat16Op after RandomRotationOp
/// Expectation: Output's shape is equal to the expected output's shape
TEST_F(MindDataTestToFloat16Op, TestOp) {
  MS_LOG(INFO) << "Doing TestRandomRotationOp::TestOp.";
  std::shared_ptr<Tensor> output_tensor;
  float s_degree = -180;
  float e_degree = 180;
  // use compute center to use for rotation
  std::vector<float> center = {};
  bool expand = false;
  auto op = std::make_unique<RandomRotationOp>(s_degree, e_degree, InterpolationMode::kLinear, expand, center, 0, 0, 0);
  EXPECT_TRUE(op->OneToOne());
  auto input_shape = input_tensor_->shape();
  EXPECT_EQ(op->Compute(input_tensor_, &output_tensor), Status::OK());
  EXPECT_EQ(input_shape.Size(), 3);
  EXPECT_EQ(output_tensor->shape().Size(), 3);
  EXPECT_EQ(input_shape[0], output_tensor->shape()[0]);
  EXPECT_EQ(input_shape[1], output_tensor->shape()[1]);

  auto to_float_op = std::make_unique<ToFloat16Op>();
  std::shared_ptr<Tensor> output_tensor1;
  EXPECT_EQ(op->Compute(output_tensor, &output_tensor1), Status::OK());
  EXPECT_EQ(output_tensor->shape()[0], output_tensor1->shape()[0]);
  EXPECT_EQ(output_tensor->shape()[1], output_tensor1->shape()[1]);
}
