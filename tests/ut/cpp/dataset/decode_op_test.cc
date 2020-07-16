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
#include <fstream>
#include "common/common.h"
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/decode_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestDecodeOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestDecodeOp() : CVOpCommon() {}
};

TEST_F(MindDataTestDecodeOp, TestOp) {
  MS_LOG(INFO) << "Doing testDecode";
  TensorShape s = TensorShape({1});
  std::shared_ptr<Tensor> output_tensor;
  DecodeOp op(true);
  op.Compute(raw_input_tensor_, &output_tensor);

  CheckImageShapeAndData(output_tensor, kDecode);
}
