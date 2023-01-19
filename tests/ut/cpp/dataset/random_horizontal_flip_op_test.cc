/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/random_horizontal_flip_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestRandomHorizontalFlipOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestRandomHorizontalFlipOp() : CVOpCommon() {}
};

/// Feature: RandomHorizontalFlip op
/// Description: Test RandomHorizontalFlipOp with prob=0.5
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestRandomHorizontalFlipOp, TestOp) {
  MS_LOG(INFO) << "Doing testHorizontalFlip.";
  // flip
  TensorRow input_tensor_row;
  std::shared_ptr<Tensor> input1;
  Tensor::CreateFromTensor(input_tensor_, &input1);
  input_tensor_row.push_back(input1);
  std::shared_ptr<Tensor> input2;
  Tensor::CreateFromTensor(input_tensor_, &input2);
  input_tensor_row.push_back(input2);
  TensorRow output_tensor_row;
  auto op = std::make_unique<RandomHorizontalFlipOp>(0.5);
  Status s = op->Compute(input_tensor_row, &output_tensor_row);
  EXPECT_TRUE(s.IsOk());
  CheckImageShapeAndData(input_tensor_, kFlipHorizontal);
  MS_LOG(INFO) << "testHorizontalFlip end.";
}

/// Feature: RandomHorizontalFlip
/// Description: process tensor with dim more than 3
/// Expectation: process successfully
TEST_F(MindDataTestRandomHorizontalFlipOp, TestVideo4DOp) {
  MS_LOG(INFO) << "Doing MindDataTestRandomHorizontalFlipOp-TestVideo4DOp.";

  TensorRow video_input_tensor_row;
  auto shape = input_tensor_->shape();
  int frame_num = 3;
  TensorShape new_shape({frame_num, shape[-3], shape[-2], shape[-1]});
  std::shared_ptr<Tensor> video_tensor_1, video_tensor_2;
  Tensor::CreateEmpty(new_shape, input_tensor_->type(), &video_tensor_1);
  Tensor::CreateEmpty(new_shape, input_tensor_->type(), &video_tensor_2);
  for (dsize_t i = 0; i < frame_num; i++) {
    video_tensor_1->InsertTensor({i}, input_tensor_, true);
    video_tensor_2->InsertTensor({i}, input_tensor_, true);
  }
  video_input_tensor_row.push_back(video_tensor_1);
  video_input_tensor_row.push_back(video_tensor_2);
  TensorRow video_output_tensor_row;
  auto video_op = std::make_shared<RandomHorizontalFlipOp>(1.0);
  Status video_s = video_op->Compute(video_input_tensor_row, &video_output_tensor_row);
  EXPECT_TRUE(video_s.IsOk());

  MS_LOG(INFO) << "TestVideo4DOp end.";
}

/// Feature: RandomHorizontalFlip
/// Description: process tensor with dim more than 3
/// Expectation: process successfully
TEST_F(MindDataTestRandomHorizontalFlipOp, TestVideo5DOp) {
  MS_LOG(INFO) << "Doing MindDataTestRandomHorizontalFlipOp-TestVideo5DOp.";

  TensorRow video_input_tensor_row;
  auto shape = input_tensor_->shape();
  int frame_num = 3;
  TensorShape new_shape({frame_num, shape[-3], shape[-2], shape[-1]});
  std::shared_ptr<Tensor> video_tensor_1;
  Tensor::CreateEmpty(new_shape, input_tensor_->type(), &video_tensor_1);
  for (dsize_t i = 0; i < frame_num; i++) {
    video_tensor_1->InsertTensor({i}, input_tensor_, true);
  }
  shape = video_tensor_1->shape();
  TensorShape five_shape({1, shape[-4], shape[-3], shape[-2], shape[-1]});
  video_tensor_1->Reshape(five_shape);
  video_input_tensor_row.push_back(video_tensor_1);
  TensorRow video_output_tensor_row;
  auto video_op = std::make_unique<RandomHorizontalFlipOp>(1.0);
  Status video_s = video_op->Compute(video_input_tensor_row, &video_output_tensor_row);
  EXPECT_TRUE(video_s.IsOk());

  MS_LOG(INFO) << "TestVideo5DOp end.";
}

/// Feature: RandomHorizontalFlip
/// Description: process tensor with dim 1
/// Expectation: expect error
TEST_F(MindDataTestRandomHorizontalFlipOp, TestVideo1DOp) {
  MS_LOG(INFO) << "Doing MindDataTestRandomHorizontalFlipOp-TestVideo1DOp.";

  TensorRow input_tensor_row;
  auto temp = input_tensor_;
  TensorShape new_shape({temp->Size()});
  temp->Reshape(new_shape);
  input_tensor_row.push_back(temp);
  TensorRow output_tensor_row;
  auto op = std::make_unique<RandomHorizontalFlipOp>(1.0);
  EXPECT_ERROR(op->Compute(input_tensor_row, &output_tensor_row));

  MS_LOG(INFO) << "TestVideo1DOp end.";
}
