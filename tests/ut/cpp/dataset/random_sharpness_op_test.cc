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

#include "minddata/dataset/kernels/image/random_sharpness_op.h"
#include "common/common.h"
#include "common/cvop_common.h"
#include "utils/log_adapter.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRandomSharpness : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomSharpness() : CVOpCommon() {}
};

TEST_F(MindDataTestRandomSharpness, TestOp) {
  MS_LOG(INFO) << "Doing test RandomSharpness.";
  // setting seed here
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(120);
  // Sharpness with a factor in range [0.2,1.8]
  float start_degree = 0.2;
  float end_degree = 1.8;
  std::shared_ptr<Tensor> output_tensor;
  // sharpening
  std::unique_ptr<RandomSharpnessOp> op(new RandomSharpnessOp(start_degree, end_degree));
  EXPECT_TRUE(op->OneToOne());
  Status st = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(st.IsOk());
  CheckImageShapeAndData(output_tensor, kRandomSharpness);
  // restoring the seed
  GlobalContext::config_manager()->set_seed(curr_seed);
  MS_LOG(INFO) << "testRandomSharpness end.";
}
