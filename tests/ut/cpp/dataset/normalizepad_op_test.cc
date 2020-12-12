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
#include "minddata/dataset/kernels/image/normalize_pad_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"
#include <opencv2/opencv.hpp>

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestNormalizePadOP : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestNormalizePadOP() : CVOpCommon() {}
};

TEST_F(MindDataTestNormalizePadOP, TestFloat32) {
  MS_LOG(INFO) << "Doing TestNormalizePadOp::TestFloat32.";
  std::shared_ptr<Tensor> output_tensor;

  // Numbers are from the resnet50 model implementation
  float mean[3] = {121.0, 115.0, 100.0};
  float std[3] = {70.0, 68.0, 71.0};

  // NormalizePad Op
  std::unique_ptr<NormalizePadOp> op(new NormalizePadOp(mean[0], mean[1], mean[2], std[0], std[1], std[2], "float32"));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
}

TEST_F(MindDataTestNormalizePadOP, TestFloat16) {
  MS_LOG(INFO) << "Doing TestNormalizePadOp::TestFloat16.";
  std::shared_ptr<Tensor> output_tensor;

  // Numbers are from the resnet50 model implementation
  float mean[3] = {121.0, 115.0, 100.0};
  float std[3] = {70.0, 68.0, 71.0};

  // NormalizePad Op
  std::unique_ptr<NormalizePadOp> op(new NormalizePadOp(mean[0], mean[1], mean[2], std[0], std[1], std[2], "float16"));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
}