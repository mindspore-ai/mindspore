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
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include <random>

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

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
      rnd_aspect_(log(aspect_lb), log(aspect_ub)),
      interpolation_(interpolation),
      aspect_lb_(aspect_lb),
      aspect_ub_(aspect_ub),
      max_iter_(max_iter) {
  rnd_.seed(GetSeed());
  is_deterministic_ = false;
}

Status RandomCropAndResizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Size() >= 2, "RandomCropAndResize: the image is not <H,W,C> or <H,W>");

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
  return Status(StatusCode::kMDUnexpectedError, "RandomCropAndResize: invalid input shape");
}
Status RandomCropAndResizeOp::GetCropBox(int h_in, int w_in, int *x, int *y, int *crop_height, int *crop_width) {
  *crop_width = w_in;
  *crop_height = h_in;
  CHECK_FAIL_RETURN_UNEXPECTED(w_in != 0, "RandomCropAndResize: Width is 0");
  CHECK_FAIL_RETURN_UNEXPECTED(h_in != 0, "RandomCropAndResize: Height is 0");
  CHECK_FAIL_RETURN_UNEXPECTED(aspect_lb_ > 0, "RandomCropAndResize: aspect lower bound must be greater than zero");
  for (int32_t i = 0; i < max_iter_; i++) {
    double const sample_scale = rnd_scale_(rnd_);
    // In case of non-symmetrical aspect ratios, use uniform distribution on a logarithmic sample_scale.
    // Note rnd_aspect_ is already a random distribution of the input aspect ratio in logarithmic sample_scale.
    double const sample_aspect = exp(rnd_aspect_(rnd_));

    *crop_width = static_cast<int32_t>(std::round(std::sqrt(h_in * w_in * sample_scale * sample_aspect)));
    *crop_height = static_cast<int32_t>(std::round(*crop_width / sample_aspect));
    if (*crop_width <= w_in && *crop_height <= h_in) {
      std::uniform_int_distribution<> rd_x(0, w_in - *crop_width);
      std::uniform_int_distribution<> rd_y(0, h_in - *crop_height);
      *x = rd_x(rnd_);
      *y = rd_y(rnd_);
      return Status::OK();
    }
  }
  double const img_aspect = static_cast<double>(w_in) / h_in;
  if (img_aspect < aspect_lb_) {
    *crop_width = w_in;
    *crop_height = static_cast<int32_t>(std::round(*crop_width / static_cast<double>(aspect_lb_)));
  } else {
    if (img_aspect > aspect_ub_) {
      *crop_height = h_in;
      *crop_width = static_cast<int32_t>(std::round(*crop_height * static_cast<double>(aspect_ub_)));
    } else {
      *crop_width = w_in;
      *crop_height = h_in;
    }
  }
  *x = static_cast<int32_t>(std::round((w_in - *crop_width) / 2.0));
  *y = static_cast<int32_t>(std::round((h_in - *crop_height) / 2.0));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
