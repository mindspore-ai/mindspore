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
#include "dataset/kernels/image/rescale_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRescaleOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRescaleOp() : CVOpCommon() {}
};

TEST_F(MindDataTestRescaleOp, TestOp) {
  // Rescale Factor
  float rescale = 1.0 / 255;
  float shift = 1.0;

  std::unique_ptr<RescaleOp> op(new RescaleOp(rescale, shift));
  std::shared_ptr<Tensor> output_tensor;
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(op->OneToOne());
  EXPECT_TRUE(s.IsOk());
  // The rescaled image becomes CV_32FC3, saving it as JPEG
  // will result in a black image since opencv converts it to int
  // This function is still good to have since it checks the shape
  // but to check the data, its better and easier to do this
  // check in python test.
  CheckImageShapeAndData(output_tensor, kRescale);
  MS_LOG(INFO) << "testRescale end.";
}
