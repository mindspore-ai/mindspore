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
#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestExecute : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestExecute() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

TEST_F(MindDataTestExecute, TestOp1) {
  MS_LOG(INFO) << "Doing testCrop.";
  // Crop params
  std::shared_ptr<TensorOperation> center_crop = vision::CenterCrop({30});
  std::shared_ptr<Tensor> out_image = Execute(std::move(center_crop))(input_tensor_);
  EXPECT_NE(out_image, nullptr);
  EXPECT_EQ(30, out_image->shape()[0]);
  EXPECT_EQ(30, out_image->shape()[1]);
}
