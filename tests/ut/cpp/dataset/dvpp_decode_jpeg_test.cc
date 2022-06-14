/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/dataset/vision_ascend.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_jpeg_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestDvppDecodeJpeg : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestDvppDecodeJpeg() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

/// Feature: DvppDecodeJpeg op
/// Description: Test DvppDecodeJpegOp basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestDvppDecodeJpeg, TestOp1) {
  MS_LOG(INFO) << "Doing testDvppDecodeJpeg.";
  auto op = std::make_unique<DvppDecodeJpegOp>();
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor_);
  EXPECT_EQ(s, Status::OK());
}
