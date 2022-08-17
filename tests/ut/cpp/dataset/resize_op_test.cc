/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/image/resize_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestResizeOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestResizeOp() : CVOpCommon() {}
};

/// Feature: Resize op
/// Description: Test ResizeOp with a factor of 0.5
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestResizeOp, TestOp) {
  MS_LOG(INFO) << "Doing testResize";
  // Resizing with a factor of 0.5
  TensorShape s = input_tensor_->shape();
  int output_w = 0.5 * s[0];
  int output_h = (s[0] * output_w) / s[1];
  std::shared_ptr<Tensor> output_tensor;
  // Resizing
  auto op = std::make_unique<ResizeOp>(output_h, output_w);
  Status st = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(st.IsOk());
  CheckImageShapeAndData(output_tensor, kResizeBilinear);
  MS_LOG(INFO) << "testResize end.";
}

/// Feature: Resize
/// Description: test Resize with 4 dimension input
/// Expectation: resize successfully
TEST_F(MindDataTestResizeOp, TestOpVideo) {
  MS_LOG(INFO) << "Doing MindDataTestResizeOp-TestOpVideo.";
  // Resizing with high dimension input
  // construct a fake 4 dimension data
  std::shared_ptr<Tensor> input_tensor_cp;
  ASSERT_OK(Tensor::CreateFromTensor(input_tensor_, &input_tensor_cp));
  std::vector<std::shared_ptr<Tensor>> tensor_list;
  tensor_list.push_back(input_tensor_cp);
  tensor_list.push_back(input_tensor_cp);

  std::shared_ptr<Tensor> input_4d;
  ASSERT_OK(TensorVectorToBatchTensor(tensor_list, &input_4d));
  TensorShape s_two = input_4d->shape();
  int output_w_two = 0.5 * s_two[1];
  int output_h_two = (s_two[1] * output_w_two) / s_two[2];
  std::shared_ptr<Tensor> output_tensor_two;
  // Resizing
  std::unique_ptr<ResizeOp> op_two = std::make_unique<ResizeOp>(output_h_two, output_w_two);
  Status st_two = op_two->Compute(input_4d, &output_tensor_two);
  EXPECT_TRUE(st_two.IsOk());

  const int HEIGHT_INDEX = -3, WIDTH_INDEX = -2;
  auto out_shape_vec = s_two.AsVector();
  auto size = out_shape_vec.size();
  out_shape_vec[size + HEIGHT_INDEX] = output_h_two;
  out_shape_vec[size + WIDTH_INDEX] = output_w_two;
  TensorShape out = TensorShape(out_shape_vec);
  EXPECT_EQ(out, output_tensor_two->shape());

  MS_LOG(INFO) << "testResize end.";
}
