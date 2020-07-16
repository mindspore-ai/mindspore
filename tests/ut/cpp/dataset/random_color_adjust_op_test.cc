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
#include "minddata/dataset/kernels/image/random_color_adjust_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRandomColorAdjustOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomColorAdjustOp() : CVOpCommon() {}
};

TEST_F(MindDataTestRandomColorAdjustOp, TestOp1) {
  MS_LOG(INFO) << "Doing testRandomColorAdjustOp.";

  std::shared_ptr<Tensor> output_tensor;
  std::unique_ptr<RandomColorAdjustOp> op(new RandomColorAdjustOp(0.7, 1.3, 0.8, 1.2, 0.8, 1.2, -0.2, 0.2));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(input_tensor_->shape()[0], output_tensor->shape()[0]);
  EXPECT_EQ(input_tensor_->shape()[1], output_tensor->shape()[1]);
}

TEST_F(MindDataTestRandomColorAdjustOp, TestOp2) {
  MS_LOG(INFO) << "Doing testRandomColorAdjustOp2.";

  std::shared_ptr<Tensor> output_tensor;
  std::unique_ptr<RandomColorAdjustOp> op(new RandomColorAdjustOp(0.7, 1.3, 0.8, 1.2, 0.8, 1.2, -0.2, 0.2));

  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(input_tensor_->shape()[0], output_tensor->shape()[0]);
  EXPECT_EQ(input_tensor_->shape()[1], output_tensor->shape()[1]);
}

TEST_F(MindDataTestRandomColorAdjustOp, TestOp3) {
  MS_LOG(INFO) << "Doing testRandomColorAdjustOp Brightness.";

  std::shared_ptr<Tensor> output_tensor;
  std::unique_ptr<RandomColorAdjustOp> op(new RandomColorAdjustOp(0.8, 0.8, 0, 0, 0, 0, 0, 0));

  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(input_tensor_->shape()[0], output_tensor->shape()[0]);
  EXPECT_EQ(input_tensor_->shape()[1], output_tensor->shape()[1]);
}

TEST_F(MindDataTestRandomColorAdjustOp, TestOp4) {
  MS_LOG(INFO) << "Doing testRandomColorAdjustOp Brightness.";

  std::shared_ptr<Tensor> output_tensor;
  std::unique_ptr<RandomColorAdjustOp> op(new RandomColorAdjustOp(0.8, 0.8, 0, 0, 0, 0, 0.2, 0.2));

  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(input_tensor_->shape()[0], output_tensor->shape()[0]);
  EXPECT_EQ(input_tensor_->shape()[1], output_tensor->shape()[1]);
}
