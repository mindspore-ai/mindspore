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
#include <random>
#include "dataset/kernels/image/random_crop_and_resize_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRandomCropAndResizeOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomCropAndResizeOp() : CVOpCommon() {}
};

TEST_F(MindDataTestRandomCropAndResizeOp, TestOpDefault) {
  MS_LOG(INFO) << "Doing testRandomCropAndResize.";
  TensorShape s_in = input_tensor_->shape();
  std::shared_ptr<Tensor> output_tensor;
  int h_out = 512;
  int w_out = 512;

  TensorShape s_out({(uint32_t) h_out, (uint32_t) w_out, (uint32_t) s_in[2]});

  std::unique_ptr<RandomCropAndResizeOp> op(new RandomCropAndResizeOp(h_out, w_out));
  Status s;
  for (auto i = 0; i < 100; i++) {
    s = op->Compute(input_tensor_, &output_tensor);
  }
  EXPECT_TRUE(s.IsOk());
  MS_LOG(INFO) << "testRandomCropAndResize end.";
}

TEST_F(MindDataTestRandomCropAndResizeOp, TestOpExtended) {
  MS_LOG(INFO) << "Doing testRandomCropAndResize.";
  TensorShape s_in = input_tensor_->shape();
  std::shared_ptr<Tensor> output_tensor;
  int h_out = 1024;
  int w_out = 2048;
  float aspect_lb = 0.2;
  float aspect_ub = 5;
  float scale_lb = 0.0001;
  float scale_ub = 1.0;

  TensorShape s_out({(uint32_t) h_out, (uint32_t) w_out, (uint32_t) s_in[2]});

  std::unique_ptr<RandomCropAndResizeOp> op(
    new RandomCropAndResizeOp(h_out, w_out, scale_lb, scale_ub, aspect_lb, aspect_ub));
  Status s;
  for (auto i = 0; i < 100; i++) {
    s = op->Compute(input_tensor_, &output_tensor);
  }
  EXPECT_TRUE(s.IsOk());
  MS_LOG(INFO) << "testRandomCropAndResize end.";
}
