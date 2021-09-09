/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/random_crop_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestRandomCropOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestRandomCropOp() : CVOpCommon() {}

  TensorRow output_tensor_row;
};

TEST_F(MindDataTestRandomCropOp, TestOp1) {
  MS_LOG(INFO) << "Doing testRandomCrop.";
  // Crop params
  unsigned int crop_height = 128;
  unsigned int crop_width = 128;
  std::unique_ptr<RandomCropOp> op(new RandomCropOp(crop_height, crop_width, 0, 0, 0, 0, false, BorderType::kConstant));
  TensorRow input_tensor_row;
  input_tensor_row.push_back(input_tensor_);
  input_tensor_row.push_back(input_tensor_);
  Status s = op->Compute(input_tensor_row, &output_tensor_row);
  for (size_t i = 0; i < input_tensor_row.size(); i++) {
    size_t actual = 0;
    if (s == Status::OK()) {
      actual = output_tensor_row[i]->shape()[0] * output_tensor_row[i]->shape()[1] * output_tensor_row[i]->shape()[2];
    }
    EXPECT_EQ(actual, crop_height * crop_width * 3);
    EXPECT_EQ(s, Status::OK());
  }
}

TEST_F(MindDataTestRandomCropOp, TestOp2) {
  MS_LOG(INFO) << "Doing testRandomCrop.";
  // Crop params
  unsigned int crop_height = 1280;
  unsigned int crop_width = 1280;
  TensorRow input_tensor_row;
  input_tensor_row.push_back(input_tensor_);
  input_tensor_row.push_back(input_tensor_);
  std::unique_ptr<RandomCropOp> op(
    new RandomCropOp(crop_height, crop_width, 513, 513, 513, 513, false, BorderType::kConstant));
  Status s = op->Compute(input_tensor_row, &output_tensor_row);
  EXPECT_EQ(true, s.IsOk());
  MS_LOG(INFO) << "testRandomCrop end.";
}
