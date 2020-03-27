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
#include "dataset/kernels/image/change_mode_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestChangeModeOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestChangeModeOp() : CVOpCommon() {}
};

TEST_F(MindDataTestChangeModeOp, TestOp) {
  MS_LOG(INFO) << "Doing MindDataTestChangeModeOp.";

  // Creating a Tensor
  TensorShape s = input_tensor_->shape();
  int size_buffer = s[0] * s[1] * s[2];

  std::unique_ptr<uchar[]> output_buffer(new uchar[size_buffer]);
  std::shared_ptr<Tensor> output_tensor(new Tensor(s, DataType(DataType::DE_UINT8)));

  std::unique_ptr<ChangeModeOp> op(new ChangeModeOp());
  op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(op->OneToOne());

  // Saving
  CheckImageShapeAndData(output_tensor, kChangeMode);

  MS_LOG(INFO) << "MindDataTestChangeModeOp end.";
}
