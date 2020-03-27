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
#include "dataset/kernels/image/distort_bounding_box_crop_op.h"
#include <random>
#include "dataset/core/cv_tensor.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/util/random.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t DistortBoundingBoxCropOp::kDefMaxAttempts = 100;
const int32_t DistortBoundingBoxCropOp::kDefBoxGenAttempts = 10;

DistortBoundingBoxCropOp::DistortBoundingBoxCropOp(float aspect_ratio, float intersect_ratio, float crop_ratio_lb,
                                                   float crop_ratio_ub, int32_t max_attempts, int32_t box_gen_attempts)
    : max_attempts_(max_attempts),
      box_gen_attempts_(box_gen_attempts),
      aspect_ratio_(aspect_ratio),
      intersect_ratio_(intersect_ratio),
      crop_ratio_lb_(crop_ratio_lb),
      crop_ratio_ub_(crop_ratio_ub) {
  seed_ = GetSeed();
  rnd_.seed(seed_);
}

Status DistortBoundingBoxCropOp::Compute(const std::vector<std::shared_ptr<Tensor>>& input,
                                         std::vector<std::shared_ptr<Tensor>>* output) {
  IO_CHECK_VECTOR(input, output);
  if (input.size() != NumInput())
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Number of inputs is not 5");

  CHECK_FAIL_RETURN_UNEXPECTED(input[1]->shape().Size() >= 1, "The shape of the second tensor is abnormal");
  int64_t num_boxes = 0;
  for (uint64_t i = 1; i < input.size(); i++) {
    if (i == 1) num_boxes = input[i]->shape()[0];
    if (num_boxes != input[i]->shape()[0])
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Numbers of boxes do not match");

    if (input[i]->type() != DataType::DE_FLOAT32)
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "boxes' type is not DE_FLOAT21");
  }

  // assume input Tensor vector in the order of [img, bbox_y_min, bbox_y_max, bbox_x_min, bbox_x_max]
  CHECK_FAIL_RETURN_UNEXPECTED(input[0]->shape().Size() >= 2, "The shape of the first tensor is abnormal");
  int h_in = input[0]->shape()[0];
  int w_in = input[0]->shape()[1];

  std::vector<cv::Rect> bounding_boxes;
  for (int64_t i = 0; i < num_boxes; ++i) {
    // bbox coordinates are floats relative to the image width and height
    float y_min, y_max, x_min, x_max;
    RETURN_IF_NOT_OK(input[1]->GetItemAt<float>(&y_min, {i}));
    RETURN_IF_NOT_OK(input[2]->GetItemAt<float>(&y_max, {i}));
    RETURN_IF_NOT_OK(input[3]->GetItemAt<float>(&x_min, {i}));
    RETURN_IF_NOT_OK(input[4]->GetItemAt<float>(&x_max, {i}));
    bounding_boxes.emplace_back(static_cast<int>(x_min * w_in), static_cast<int>(y_min * h_in),
                                static_cast<int>((x_max - x_min) * w_in), static_cast<int>((y_max - y_min) * h_in));
  }
  cv::Rect output_box;
  bool should_crop = false;

  // go over iterations, if no satisfying box found we return the original image
  for (int32_t t = 0; t < max_attempts_; ++t) {
    // try to generate random box
    RETURN_IF_NOT_OK(GenerateRandomCropBox(h_in, w_in, aspect_ratio_, crop_ratio_lb_, crop_ratio_ub_,
                                           box_gen_attempts_,  // int maxIter,  should not be needed here
                                           &output_box, seed_));
    RETURN_IF_NOT_OK(CheckOverlapConstraint(output_box,
                                            bounding_boxes,  // have to change, should take tensor or add bbox logic
                                            intersect_ratio_, &should_crop));
    if (should_crop) {
      // found a box to crop
      break;
    }
  }
  // essentially we have to check this again at the end to return original tensor
  if (should_crop) {
    std::shared_ptr<Tensor> out;
    RETURN_IF_NOT_OK(Crop(input[0], &out, output_box.x, output_box.y, output_box.width, output_box.height));
    output->push_back(out);
  } else {
    output->push_back(input[0]);
  }
  return Status::OK();
}

Status DistortBoundingBoxCropOp::OutputShape(const std::vector<TensorShape>& inputs,
                                             std::vector<TensorShape>& outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out = TensorShape{-1, -1};
  if (inputs[0].Rank() == 2) outputs.emplace_back(out);
  if (inputs[0].Rank() == 3) outputs.emplace_back(out.AppendDim(inputs[0][2]));
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kUnexpectedError, "Input has a wrong shape");
}
Status DistortBoundingBoxCropOp::OutputType(const std::vector<DataType>& inputs, std::vector<DataType>& outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  outputs[0] = inputs[0];
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
