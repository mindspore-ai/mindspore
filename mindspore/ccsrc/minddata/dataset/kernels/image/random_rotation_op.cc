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
#include "minddata/dataset/kernels/image/random_rotation_op.h"

#include <random>

#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const float RandomRotationOp::kDefCenterX = -1;
const float RandomRotationOp::kDefCenterY = -1;
const InterpolationMode RandomRotationOp::kDefInterpolation = InterpolationMode::kNearestNeighbour;
const bool RandomRotationOp::kDefExpand = false;
const uint8_t RandomRotationOp::kDefFillR = 0;
const uint8_t RandomRotationOp::kDefFillG = 0;
const uint8_t RandomRotationOp::kDefFillB = 0;

// constructor
RandomRotationOp::RandomRotationOp(float start_degree, float end_degree, float center_x, float center_y,
                                   InterpolationMode interpolation, bool expand, uint8_t fill_r, uint8_t fill_g,
                                   uint8_t fill_b)
    : degree_start_(start_degree),
      degree_end_(end_degree),
      center_x_(center_x),
      center_y_(center_y),
      interpolation_(interpolation),
      expand_(expand),
      fill_r_(fill_r),
      fill_g_(fill_g),
      fill_b_(fill_b) {
  rnd_.seed(GetSeed());
  is_deterministic_ = false;
}

// main function call for random rotation : Generate the random degrees
Status RandomRotationOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  float random_double = distribution_(rnd_);
  // get the degree rotation range, mod by 360 because full rotation doesn't affect
  // the way this op works (uniform distribution)
  // assumption here is that mDegreesEnd > mDegreeStart so we always get positive number
  // Note: the range technically is greater than 360 degrees, but will be halved
  float degree_range = (degree_end_ - degree_start_) / 2;
  float mid = (degree_end_ + degree_start_) / 2;
  float degree = mid + random_double * degree_range;

  return Rotate(input, output, center_x_, center_y_, degree, interpolation_, expand_, fill_r_, fill_g_, fill_b_);
}
Status RandomRotationOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
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
  return Status(StatusCode::kMDUnexpectedError, "RandomRotation: invalid input shape.");
}
}  // namespace dataset
}  // namespace mindspore
