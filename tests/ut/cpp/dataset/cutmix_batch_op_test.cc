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
#include "common/common.h"
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/cutmix_batch_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestCutMixBatchOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestCutMixBatchOp() : CVOpCommon() {}
};

/// Feature: CutMixBatch op
/// Description: Test CutMixBatch op with alpha=1.0 and prob=1.0
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestCutMixBatchOp, TestSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestCutMixBatchOp success1 case";
  std::shared_ptr<Tensor> input_tensor_resized;
  std::shared_ptr<Tensor> batched_tensor;
  std::shared_ptr<Tensor> batched_labels;
  Resize(input_tensor_, &input_tensor_resized, 227, 403);

  Tensor::CreateEmpty(TensorShape({2, input_tensor_resized->shape()[0], input_tensor_resized->shape()[1],
                      input_tensor_resized->shape()[2]}), input_tensor_resized->type(), &batched_tensor);
  for (int i = 0; i < 2; i++) {
    batched_tensor->InsertTensor({i}, input_tensor_resized);
  }
  Tensor::CreateFromVector(std::vector<uint32_t>({0, 1, 1, 0}), TensorShape({2, 2}), &batched_labels);
  std::shared_ptr<CutMixBatchOp> op = std::make_shared<CutMixBatchOp>(ImageBatchFormat::kNHWC, 1.0, 1.0);
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

/// Feature: CutMixBatch op
/// Description: Test CutMixBatch op with alpha=1.0 and prob=0.5
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestCutMixBatchOp, TestSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestCutMixBatchOp success2 case";
  std::shared_ptr<Tensor> input_tensor_resized;
  std::shared_ptr<Tensor> batched_tensor;
  std::shared_ptr<Tensor> batched_labels;
  std::shared_ptr<Tensor> chw_tensor;
  Resize(input_tensor_, &input_tensor_resized, 227, 403);

  ASSERT_TRUE(HwcToChw(input_tensor_resized, &chw_tensor).IsOk());
  Tensor::CreateEmpty(TensorShape({2, chw_tensor->shape()[0], chw_tensor->shape()[1], chw_tensor->shape()[2]}),
                      chw_tensor->type(), &batched_tensor);
  for (int i = 0; i < 2; i++) {
    batched_tensor->InsertTensor({i}, chw_tensor);
  }
  Tensor::CreateFromVector(std::vector<uint32_t>({0, 1, 1, 0}), TensorShape({2, 2}), &batched_labels);
  std::shared_ptr<CutMixBatchOp> op = std::make_shared<CutMixBatchOp>(ImageBatchFormat::kNCHW, 1.0, 0.5);
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

/// Feature: CutMixBatch op
/// Description: Test CutMixBatch op where labels are not batched and are 1-dimensional
/// Expectation: Throw correct error and message
TEST_F(MindDataTestCutMixBatchOp, TestFail1) {
  // This is a fail case because our labels are not batched and are 1-dimensional
  MS_LOG(INFO) << "Doing MindDataTestCutMixBatchOp fail1 case";
  std::shared_ptr<Tensor> labels;
  Tensor::CreateFromVector(std::vector<uint32_t>({0, 1, 1, 0}), TensorShape({4}), &labels);
  std::shared_ptr<CutMixBatchOp> op = std::make_shared<CutMixBatchOp>(ImageBatchFormat::kNHWC, 1.0, 1.0);
  TensorRow in;
  in.push_back(input_tensor_);
  in.push_back(labels);
  TensorRow out;
  ASSERT_FALSE(op->Compute(in, &out).IsOk());
}

/// Feature: CutMixBatch op
/// Description: Test CutMixBatch op where image_batch_format provided
///     is not the same as the actual format of the images
/// Expectation: Throw correct error and message
TEST_F(MindDataTestCutMixBatchOp, TestFail2) {
  // This should fail because the image_batch_format provided is not the same as the actual format of the images
  MS_LOG(INFO) << "Doing MindDataTestCutMixBatchOp fail2 case";
  std::shared_ptr<Tensor> batched_tensor;
  std::shared_ptr<Tensor> batched_labels;
  Tensor::CreateEmpty(TensorShape({2, input_tensor_->shape()[0], input_tensor_->shape()[1], input_tensor_->shape()[2]}),
                      input_tensor_->type(), &batched_tensor);
  for (int i = 0; i < 2; i++) {
    std::shared_ptr<Tensor> input1;
    Tensor::CreateFromTensor(input_tensor_, &input1);
    batched_tensor->InsertTensor({i}, input1);
  }
  Tensor::CreateFromVector(std::vector<uint32_t>({0, 1, 1, 0}), TensorShape({2, 2}), &batched_labels);
  std::shared_ptr<CutMixBatchOp> op = std::make_shared<CutMixBatchOp>(ImageBatchFormat::kNCHW, 1.0, 1.0);
  TensorRow in;
  in.push_back(batched_tensor);
  in.push_back(batched_labels);
  TensorRow out;
  ASSERT_FALSE(op->Compute(in, &out).IsOk());
}
