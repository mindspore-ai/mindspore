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
#include "dataset/kernels/image/random_horizontal_flip_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRandomHorizontalFlipOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestRandomHorizontalFlipOp() : CVOpCommon() {}
};

TEST_F(MindDataTestRandomHorizontalFlipOp, TestOp) {
  MS_LOG(INFO) << "Doing testHorizontalFlip.";
  // flip
  std::unique_ptr<RandomHorizontalFlipOp> op(new RandomHorizontalFlipOp(0.5));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &input_tensor_);
  EXPECT_TRUE(s.IsOk());
  CheckImageShapeAndData(input_tensor_, kFlipHorizontal);
  MS_LOG(INFO) << "testHorizontalFlip end.";
}
