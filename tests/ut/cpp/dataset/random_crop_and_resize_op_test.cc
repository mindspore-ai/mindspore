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
#include <random>

#include "common/common.h"
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestRandomCropAndResizeOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomCropAndResizeOp() : CVOpCommon() {}
};

/// Feature: RandomCropAndResize op
/// Description: Test RandomCropAndResizeOp with aspect_lb=2, aspect_ub=2.5, scale_lb=0.2, and scale_ub=2.0
/// Expectation: Runs successfully
TEST_F(MindDataTestRandomCropAndResizeOp, TestOpSimpleTest1) {
  MS_LOG(INFO) << " starting RandomCropAndResizeOp simple test";
  TensorShape s_in = input_tensor_->shape();
  TensorRow output_tensor_row;
  std::shared_ptr<Tensor> output_tensor;
  TensorShape s_out({1024, 2048, s_in[2]});

  auto op = std::make_unique<RandomCropAndResizeOp>(1024, 2048, 0.2, 2.0, 2.0, 2.5, InterpolationMode::kLinear, 10);
  Status s;
  for (auto i = 0; i < 100; i++) {
    TensorRow input_tensor_row;
    std::shared_ptr<Tensor> input1;
    Tensor::CreateFromTensor(input_tensor_, &input1);
    input_tensor_row.push_back(input1);
    std::shared_ptr<Tensor> input2;
    Tensor::CreateFromTensor(input_tensor_, &input2);
    input_tensor_row.push_back(input2);
    s = op->Compute(input_tensor_row, &output_tensor_row);
    EXPECT_TRUE(s.IsOk());
  }

  MS_LOG(INFO) << "RandomCropAndResizeOp simple test finished";
}

/// Feature: RandomCropAndResize op
/// Description: Test RandomCropAndResizeOp with aspect_lb=1, aspect_ub=1.5, scale_lb=0.2, and scale_ub=2.0
/// Expectation: Runs successfully
TEST_F(MindDataTestRandomCropAndResizeOp, TestOpSimpleTest2) {
  MS_LOG(INFO) << " starting RandomCropAndResizeOp simple test";
  TensorShape s_in = input_tensor_->shape();
  TensorRow output_tensor_row;
  std::shared_ptr<Tensor> output_tensor;
  TensorShape s_out({1024, 2048, s_in[2]});

  auto op = std::make_unique<RandomCropAndResizeOp>(1024, 2048, 0.2, 2.0, 1.0, 1.5, InterpolationMode::kLinear, 10);
  Status s;
  for (auto i = 0; i < 100; i++) {
    TensorRow input_tensor_row;
    std::shared_ptr<Tensor> input1;
    Tensor::CreateFromTensor(input_tensor_, &input1);
    input_tensor_row.push_back(input1);
    std::shared_ptr<Tensor> input2;
    Tensor::CreateFromTensor(input_tensor_, &input2);
    input_tensor_row.push_back(input2);
    s = op->Compute(input_tensor_row, &output_tensor_row);
    EXPECT_TRUE(s.IsOk());
  }

  MS_LOG(INFO) << "RandomCropAndResizeOp simple test finished";
}

/// Feature: RandomCropAndResize op
/// Description: Test RandomCropAndResizeOp with aspect_lb=0.2, aspect_ub=3, scale_lb=0.2, and scale_ub=2.0
/// Expectation: Runs successfully
TEST_F(MindDataTestRandomCropAndResizeOp, TestOpSimpleTest3) {
  MS_LOG(INFO) << " starting RandomCropAndResizeOp simple test";
  TensorShape s_in = input_tensor_->shape();
  TensorRow output_tensor_row;
  std::shared_ptr<Tensor> output_tensor;
  TensorShape s_out({1024, 2048, s_in[2]});

  auto op = std::make_unique<RandomCropAndResizeOp>(1024, 2048, 0.2, 2.0, 0.2, 3.0, InterpolationMode::kLinear, 10);
  Status s;
  for (auto i = 0; i < 100; i++) {
    TensorRow input_tensor_row;
    std::shared_ptr<Tensor> input1;
    Tensor::CreateFromTensor(input_tensor_, &input1);
    input_tensor_row.push_back(input1);
    std::shared_ptr<Tensor> input2;
    Tensor::CreateFromTensor(input_tensor_, &input2);
    input_tensor_row.push_back(input2);
    s = op->Compute(input_tensor_row, &output_tensor_row);
    EXPECT_TRUE(s.IsOk());
  }

  MS_LOG(INFO) << "RandomCropAndResizeOp simple test finished";
}
