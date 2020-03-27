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
#include "dataset/kernels/image/random_crop_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRandomCropOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestRandomCropOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

TEST_F(MindDataTestRandomCropOp, TestOp1) {
  MS_LOG(INFO) << "Doing testRandomCrop.";
  // Crop params
  unsigned int crop_height = 128;
  unsigned int crop_width = 128;
  std::unique_ptr<RandomCropOp> op(new RandomCropOp(crop_height, crop_width, 0, 0, 0, 0, BorderType::kConstant, false));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor_);
  size_t actual = 0;
  if (s == Status::OK()) {
    actual = output_tensor_->shape()[0] * output_tensor_->shape()[1] * output_tensor_->shape()[2];
  }
  EXPECT_EQ(actual, crop_height * crop_width * 3);
  EXPECT_EQ(s, Status::OK());
}

TEST_F(MindDataTestRandomCropOp, TestOp2) {
  MS_LOG(INFO) << "Doing testRandomCrop.";
  // Crop params
  unsigned int crop_height = 1280;
  unsigned int crop_width = 1280;
  std::unique_ptr<RandomCropOp> op(
    new RandomCropOp(crop_height, crop_width, 513, 513, 513, 513, BorderType::kConstant, false));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor_);
  EXPECT_EQ(true, s.IsOk());
  MS_LOG(INFO) << "testRandomCrop end.";
}
