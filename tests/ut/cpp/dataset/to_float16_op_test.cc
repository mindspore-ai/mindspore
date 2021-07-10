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
#include "minddata/dataset/kernels/image/random_rotation_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/data/to_float16_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestToFloat16Op : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestToFloat16Op() : CVOpCommon() {}
};

TEST_F(MindDataTestToFloat16Op, TestOp) {
  MS_LOG(INFO) << "Doing TestRandomRotationOp::TestOp.";
  std::shared_ptr<Tensor> output_tensor;
  float s_degree = -180;
  float e_degree = 180;
  // use compute center to use for rotation
  std::vector<float> center = {};
  bool expand = false;
  std::unique_ptr<RandomRotationOp> op(
    new RandomRotationOp(s_degree, e_degree, InterpolationMode::kLinear, expand, center));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(input_tensor_->shape()[0], output_tensor->shape()[0]);
  EXPECT_EQ(input_tensor_->shape()[1], output_tensor->shape()[1]);

  std::unique_ptr<ToFloat16Op> to_float_op(new ToFloat16Op());
  std::shared_ptr<Tensor> output_tensor1;
  s = op->Compute(output_tensor, &output_tensor1);
  EXPECT_EQ(output_tensor->shape()[0], output_tensor1->shape()[0]);
  EXPECT_EQ(output_tensor->shape()[1], output_tensor1->shape()[1]);
}
