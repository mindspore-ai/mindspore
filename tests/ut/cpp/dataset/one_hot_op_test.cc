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
#include "minddata/dataset/kernels/data/one_hot_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestOneHotOp : public UT::Common {
 protected:
    MindDataTestOneHotOp() {}
};

TEST_F(MindDataTestOneHotOp, TestOp) {
  MS_LOG(INFO) << "Doing MindDataTestOneHotOp.";
  std::vector<uint64_t> labels = {0, 1, 2};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, &input);
  std::shared_ptr<Tensor> output;

  std::unique_ptr<OneHotOp> op(new OneHotOp(5));
  Status s = op->Compute(input, &output);
  std::vector<uint64_t> out = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape{3, 5}, &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());
  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestOneHotOp end.";
}
