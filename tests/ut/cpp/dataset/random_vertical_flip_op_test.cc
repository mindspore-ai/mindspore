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
#include "dataset/kernels/image/random_vertical_flip_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRandomVerticalFlipOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestRandomVerticalFlipOp() : CVOpCommon() {}
};

TEST_F(MindDataTestRandomVerticalFlipOp, TestOp) {
  MS_LOG(INFO) << "Doing testVerticalFlip.";
  // flip
  std::unique_ptr<RandomVerticalFlipOp> op(new RandomVerticalFlipOp(0.5));
  Status s = op->Compute(input_tensor_, &input_tensor_);
  EXPECT_TRUE(op->OneToOne());
  EXPECT_TRUE(s.IsOk());
  CheckImageShapeAndData(input_tensor_, kFlipVertical);
  MS_LOG(INFO) << "testVerticalFlip end.";
}
