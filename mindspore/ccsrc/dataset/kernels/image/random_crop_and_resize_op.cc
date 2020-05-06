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
#include "dataset/kernels/image/random_crop_and_resize_op.h"
#include <random>

#include "dataset/kernels/image/image_utils.h"
#include "dataset/util/random.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
const float RandomCropAndResizeOp::kDefScaleLb = 0.08;
const float RandomCropAndResizeOp::kDefScaleUb = 1.0;
const float RandomCropAndResizeOp::kDefAspectLb = 0.75;
const float RandomCropAndResizeOp::kDefAspectUb = 1.333333;
const InterpolationMode RandomCropAndResizeOp::kDefInterpolation = InterpolationMode::kLinear;
const int32_t RandomCropAndResizeOp::kDefMaxIter = 10;

RandomCropAndResizeOp::RandomCropAndResizeOp(int32_t target_height, int32_t target_width, float scale_lb,
                                             float scale_ub, float aspect_lb, float aspect_ub,
                                             InterpolationMode interpolation, int32_t max_iter)
    : target_height_(target_height),
      target_width_(target_width),
      rnd_scale_(scale_lb, scale_ub),
      rnd_aspect_(aspect_lb, aspect_ub),
      interpolation_(interpolation),
      max_iter_(max_iter) {
  rnd_.seed(GetSeed());
}

Status RandomCropAndResizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Size() >= 2, "The shape of input is abnormal");

  int h_in = input->shape()[0];
  int w_in = input->shape()[1];
  int x = 0;
  int y = 0;
  int crop_height = 0;
  int crop_width = 0;
  (void)GetCropBox(h_in, w_in, &x, &y, &crop_height, &crop_width);
  return CropAndResize(input, output, x, y, crop_height, crop_width, target_height_, target_width_, interpolation_);
}
Status RandomCropAndResizeOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out = TensorShape{target_height_, target_width_};
  if (inputs[0].Rank() == 2) outputs.emplace_back(out);
  if (inputs[0].Rank() == 3) outputs.emplace_back(out.AppendDim(inputs[0][2]));
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kUnexpectedError, "Input has a wrong shape");
}
Status RandomCropAndResizeOp::GetCropBox(int h_in, int w_in, int *x, int *y, int *crop_height, int *crop_width) {
  double scale, aspect;
  *crop_width = w_in;
  *crop_height = h_in;
  bool crop_success = false;
  for (int32_t i = 0; i < max_iter_; i++) {
    scale = rnd_scale_(rnd_);
    aspect = rnd_aspect_(rnd_);
    *crop_width = static_cast<int32_t>(std::round(std::sqrt(h_in * w_in * scale / aspect)));
    *crop_height = static_cast<int32_t>(std::round(*crop_width * aspect));
    if (*crop_width <= w_in && *crop_height <= h_in) {
      crop_success = true;
      break;
    }
  }
  if (!crop_success) {
    CHECK_FAIL_RETURN_UNEXPECTED(w_in != 0, "Width is 0");
    aspect = static_cast<double>(h_in) / w_in;
    scale = rnd_scale_(rnd_);
    *crop_width = static_cast<int32_t>(std::round(std::sqrt(h_in * w_in * scale / aspect)));
    *crop_height = static_cast<int32_t>(std::round(*crop_width * aspect));
    *crop_height = (*crop_height > h_in) ? h_in : *crop_height;
    *crop_width = (*crop_width > w_in) ? w_in : *crop_width;
  }
  std::uniform_int_distribution<> rd_x(0, w_in - *crop_width);
  std::uniform_int_distribution<> rd_y(0, h_in - *crop_height);
  *x = rd_x(rnd_);
  *y = rd_y(rnd_);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
