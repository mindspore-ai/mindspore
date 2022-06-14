/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/invert_op.h"
#include "common/common.h"
#include "common/cvop_common.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestInvert : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestInvert() : CVOpCommon() {}
};

/// Feature: Invert op
/// Description: Test Invert op basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestInvert, TestOp) {
  MS_LOG(INFO) << "Doing test Invert.";
  std::shared_ptr<Tensor> output_tensor;
  auto op = std::make_unique<InvertOp>();
  EXPECT_TRUE(op->OneToOne());
  Status st = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(st.IsOk());
  CheckImageShapeAndData(output_tensor, kInvert);
  MS_LOG(INFO) << "testInvert end.";
}
