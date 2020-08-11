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
#include "minddata/dataset/kernels/image/mixup_batch_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestMixUpBatchOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestMixUpBatchOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

TEST_F(MindDataTestMixUpBatchOp, TestSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestMixUpBatchOp success case";
  std::shared_ptr<Tensor> batched_tensor;
  std::shared_ptr<Tensor> batched_labels;
  Tensor::CreateEmpty(TensorShape({2, input_tensor_->shape()[0], input_tensor_->shape()[1], input_tensor_->shape()[2]}), input_tensor_->type(), &batched_tensor);
  for (int i = 0; i < 2; i++) {
  batched_tensor->InsertTensor({i}, input_tensor_);
  }
  Tensor::CreateFromVector(std::vector<uint32_t>({0, 1, 1, 0}), TensorShape({2, 2}), &batched_labels);
  std::shared_ptr<MixUpBatchOp> op = std::make_shared<MixUpBatchOp>(1);
  TensorRow in;
  in.push_back(batched_tensor);
  in.push_back(batched_labels);
  TensorRow out;
  ASSERT_TRUE(op->Compute(in, &out).IsOk());

  EXPECT_EQ(in.at(0)->shape()[0], out.at(0)->shape()[0]);
  EXPECT_EQ(in.at(0)->shape()[1], out.at(0)->shape()[1]);
  EXPECT_EQ(in.at(0)->shape()[2], out.at(0)->shape()[2]);
  EXPECT_EQ(in.at(0)->shape()[3], out.at(0)->shape()[3]);

  EXPECT_EQ(in.at(1)->shape()[0], out.at(1)->shape()[0]);
  EXPECT_EQ(in.at(1)->shape()[1], out.at(1)->shape()[1]);
}

TEST_F(MindDataTestMixUpBatchOp, TestFail) {
  // This is a fail case because our labels are not batched and are 1-dimensional
  MS_LOG(INFO) << "Doing MindDataTestMixUpBatchOp fail case";
  std::shared_ptr<Tensor> labels;
  Tensor::CreateFromVector(std::vector<uint32_t>({0, 1, 1, 0}), TensorShape({4}), &labels);
  std::shared_ptr<MixUpBatchOp> op = std::make_shared<MixUpBatchOp>(1);
  TensorRow in;
  in.push_back(input_tensor_);
  in.push_back(labels);
  TensorRow out;
  ASSERT_FALSE(op->Compute(in, &out).IsOk());
}
