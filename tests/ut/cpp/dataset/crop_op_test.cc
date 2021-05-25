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
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/crop_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestCropOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestCropOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

TEST_F(MindDataTestCropOp, TestOp1) {
  MS_LOG(INFO) << "Doing testCrop.";
  // Crop params
  int crop_height = 18;
  int crop_width = 12;
  std::unique_ptr<CropOp> op(new CropOp(0, 0, crop_height, crop_width));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor_);
  size_t actual = 0;
  if (s == Status::OK()) {
    actual = output_tensor_->shape()[0] * output_tensor_->shape()[1] * output_tensor_->shape()[2];
  }
  EXPECT_EQ(crop_height, output_tensor_->shape()[0]);
  EXPECT_EQ(actual, crop_height * crop_width * 3);
  EXPECT_EQ(s, Status::OK());
}

TEST_F(MindDataTestCropOp, TestOp2) {
  MS_LOG(INFO) << "Doing testCrop negative coordinates.";
  // Crop params
  unsigned int crop_height = 10;
  unsigned int crop_width = 10;

  std::unique_ptr<CropOp> op(new CropOp(-10, -10, crop_height, crop_width));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor_);
  EXPECT_EQ(false, s.IsOk());
  MS_LOG(INFO) << "testCrop coordinate exception end.";
}

TEST_F(MindDataTestCropOp, TestOp3) {
  MS_LOG(INFO) << "Doing testCrop size too large.";
  // Crop params
  unsigned int crop_height = 1200000;
  unsigned int crop_width = 1200000;

  std::unique_ptr<CropOp> op(new CropOp(0, 0, crop_height, crop_width));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor_);
  EXPECT_EQ(false, s.IsOk());
  MS_LOG(INFO) << "testCrop size exception end.";
}
