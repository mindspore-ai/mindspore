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
#include "minddata/dataset/kernels/image/random_resize_op.h"
#include "common/common.h"
#include "common/cvop_common.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestRandomResize : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomResize() : CVOpCommon() {}
};

TEST_F(MindDataTestRandomResize, TestOp) {
  MS_LOG(INFO) << "Doing test RandomResize.";
  // Resizing with a factor of 0.5
  TensorShape s = input_tensor_->shape();
  int output_h = 0.5 * s[0];
  int output_w = 0.5 * s[1];
  TensorRow input_tensor_row;
  input_tensor_row.push_back(input_tensor_);
  input_tensor_row.push_back(input_tensor_);
  TensorRow output_tensor_row;
  // Resizing
  std::unique_ptr<RandomResizeOp> op(new RandomResizeOp(output_h, output_w));
  Status st = op->Compute(input_tensor_row, &output_tensor_row);
  EXPECT_TRUE(st.IsOk());
  MS_LOG(INFO) << "testResize end.";
}
