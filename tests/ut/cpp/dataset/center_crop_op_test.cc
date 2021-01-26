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
#include "minddata/dataset/kernels/image/center_crop_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestCenterCropOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestCenterCropOp() : CVOpCommon() {}
};

TEST_F(MindDataTestCenterCropOp, TestOp1) {
  MS_LOG(INFO) << "Doing MindDataTestCenterCropOp::TestOp1.";
  std::shared_ptr<Tensor> output_tensor;
  int het = 256;
  int wid = 128;
  std::unique_ptr<CenterCropOp> op(new CenterCropOp(het, wid));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(het, output_tensor->shape()[0]);
  EXPECT_EQ(wid, output_tensor->shape()[1]);
  std::shared_ptr<CVTensor> p = CVTensor::AsCVTensor(output_tensor);
}

TEST_F(MindDataTestCenterCropOp, TestOp2) {
  MS_LOG(INFO) << "MindDataTestCenterCropOp::TestOp2. Cap valid crop size at 10 times the input size";
  std::shared_ptr<Tensor> output_tensor;

  int64_t wid = input_tensor_->shape()[0] * 10 + 1;
  int64_t het = input_tensor_->shape()[1] * 10 + 1;

  std::unique_ptr<CenterCropOp> op(new CenterCropOp(het, wid));
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsError());
  ASSERT_TRUE(s.StatusCode() == StatusCode::kMDUnexpectedError);
}

TEST_F(MindDataTestCenterCropOp, TestOp3) {
  MS_LOG(INFO) << "Doing MindDataTestCenterCropOp::TestOp3. Test single integer input for square crop.";
  std::shared_ptr<Tensor> output_tensor;
  int side = 128;
  std::unique_ptr<CenterCropOp> op(new CenterCropOp(side));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  // Confirm both height and width are of size <side>.
  EXPECT_EQ(side, output_tensor->shape()[0]);
  EXPECT_EQ(side, output_tensor->shape()[1]);
  std::shared_ptr<CVTensor> p = CVTensor::AsCVTensor(output_tensor);
}
