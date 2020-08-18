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
#include "minddata/dataset/kernels/image/random_solarize_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/util/status.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestRandomSolarizeOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestRandomSolarizeOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

TEST_F(MindDataTestRandomSolarizeOp, TestOp1) {
  MS_LOG(INFO) << "Doing testRandomSolarizeOp1.";
  // setting seed here
  uint32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(0);

  std::vector<uint8_t> threshold = {100, 100};
  std::unique_ptr<RandomSolarizeOp> op(new RandomSolarizeOp(threshold));

  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor_);

  EXPECT_TRUE(s.IsOk());
  CheckImageShapeAndData(output_tensor_, kRandomSolarize);
  // restoring the seed
  GlobalContext::config_manager()->set_seed(curr_seed);
}