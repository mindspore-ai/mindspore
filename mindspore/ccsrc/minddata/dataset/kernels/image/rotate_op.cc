/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/rotate_op.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif

namespace mindspore {
namespace dataset {
const std::vector<float> RotateOp::kDefCenter = {};
const InterpolationMode RotateOp::kDefInterpolation = InterpolationMode::kNearestNeighbour;
const bool RotateOp::kDefExpand = false;
const uint8_t RotateOp::kDefFillR = 0;
const uint8_t RotateOp::kDefFillG = 0;
const uint8_t RotateOp::kDefFillB = 0;

RotateOp::RotateOp(int angle_id)
    : angle_id_(angle_id),
      degrees_(0),
      center_({}),
      interpolation_(InterpolationMode::kLinear),
      expand_(false),
      fill_r_(0),
      fill_g_(0),
      fill_b_(0) {}

RotateOp::RotateOp(float degrees, InterpolationMode resample, bool expand, std::vector<float> center, uint8_t fill_r,
                   uint8_t fill_g, uint8_t fill_b)
    : angle_id_(0),
      degrees_(degrees),
      center_(center),
      interpolation_(resample),
      expand_(expand),
      fill_r_(fill_r),
      fill_g_(fill_g),
      fill_b_(fill_b) {}

Status RotateOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImageRank("Rotate", input->shape().Size()));
#ifndef ENABLE_ANDROID
  return Rotate(input, output, center_, degrees_, interpolation_, expand_, fill_r_, fill_g_, fill_b_);
#else
  return Rotate(input, output, angle_id_);
#endif
}

Status RotateOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
#ifndef ENABLE_ANDROID
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  int32_t outputH = -1, outputW = -1;
  // if expand_, then we cannot know the shape. We need the input image to find the output shape --> set it to
  // <-1,-1[,3]>
  if (!expand_) {
    outputH = inputs[0][0];
    outputW = inputs[0][1];
  }
  TensorShape out = TensorShape{outputH, outputW};
  if (inputs[0].Rank() == 2) outputs.emplace_back(out);
  if (inputs[0].Rank() == 3) outputs.emplace_back(out.AppendDim(inputs[0][2]));
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kMDUnexpectedError,
                "Rotate: invalid input shape, expected 2D or 3D input, but got input dimension is:" +
                  std::to_string(inputs[0].Rank()));
#else
  if (inputs.size() != NumInput())
    return Status(StatusCode::kMDUnexpectedError,
                  "The size of the input argument vector: " + std::to_string(inputs.size()) +
                    ", does not match the number of inputs: " + std::to_string(NumInput()));
  outputs = inputs;
  return Status::OK();
#endif
}
}  // namespace dataset
}  // namespace mindspore
