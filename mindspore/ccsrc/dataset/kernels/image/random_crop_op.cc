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
#include "dataset/kernels/image/random_crop_op.h"
#include <random>
#include "dataset/kernels/image/image_utils.h"
#include "dataset/util/random.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t RandomCropOp::kDefPadTop = 0;
const int32_t RandomCropOp::kDefPadBottom = 0;
const int32_t RandomCropOp::kDefPadLeft = 0;
const int32_t RandomCropOp::kDefPadRight = 0;
const BorderType RandomCropOp::kDefBorderType = BorderType::kConstant;
const bool RandomCropOp::kDefPadIfNeeded = false;
const uint8_t RandomCropOp::kDefFillR = 0;
const uint8_t RandomCropOp::kDefFillG = 0;
const uint8_t RandomCropOp::kDefFillB = 0;

RandomCropOp::RandomCropOp(int32_t crop_height, int32_t crop_width, int32_t pad_top, int32_t pad_bottom,
                           int32_t pad_left, int32_t pad_right, BorderType border_types, bool pad_if_needed,
                           uint8_t fill_r, uint8_t fill_g, uint8_t fill_b)
    : crop_height_(crop_height),
      crop_width_(crop_width),
      pad_top_(pad_top),
      pad_bottom_(pad_bottom),
      pad_left_(pad_left),
      pad_right_(pad_right),
      pad_if_needed_(pad_if_needed),
      border_type_(border_types),
      fill_r_(fill_r),
      fill_g_(fill_g),
      fill_b_(fill_b) {
  rnd_.seed(GetSeed());
}

Status RandomCropOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  // Apply padding first then crop
  std::shared_ptr<Tensor> pad_image;

  RETURN_IF_NOT_OK(
    Pad(input, &pad_image, pad_top_, pad_bottom_, pad_left_, pad_right_, border_type_, fill_r_, fill_g_, fill_b_));
  CHECK_FAIL_RETURN_UNEXPECTED(pad_image->shape().Size() >= 2, "Abnormal shape");
  int32_t padded_image_h = pad_image->shape()[0];
  int32_t padded_image_w = pad_image->shape()[1];
  // no need to crop if same  size
  if (padded_image_h == crop_height_ && padded_image_w == crop_width_) {
    *output = pad_image;
    return Status::OK();
  }
  if (pad_if_needed_) {
    // check the dimensions of the image for padding, if we do need padding, then we change the pad values
    if (padded_image_h < crop_height_) {
      RETURN_IF_NOT_OK(Pad(pad_image, &pad_image, crop_height_ - padded_image_h, crop_height_ - padded_image_h, 0, 0,
                           border_type_, fill_r_, fill_g_, fill_b_));
    }
    if (padded_image_w < crop_width_) {
      RETURN_IF_NOT_OK(Pad(pad_image, &pad_image, 0, 0, crop_width_ - padded_image_w, crop_width_ - padded_image_w,
                           border_type_, fill_r_, fill_g_, fill_b_));
    }
    padded_image_h = pad_image->shape()[0];
    padded_image_w = pad_image->shape()[1];
  }
  if (padded_image_h < crop_height_ || padded_image_w < crop_width_ || crop_height_ == 0 || crop_width_ == 0) {
    return Status(StatusCode::kShapeMisMatch, __LINE__, __FILE__,
                  "Crop size is greater than the image dimensions or is zero.");
  }
  // random top corner
  int x = std::uniform_int_distribution<int>(0, padded_image_w - crop_width_)(rnd_);
  int y = std::uniform_int_distribution<int>(0, padded_image_h - crop_height_)(rnd_);
  return Crop(pad_image, output, x, y, crop_width_, crop_height_);
}
Status RandomCropOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out = TensorShape{crop_height_, crop_width_};
  if (inputs[0].Rank() == 2) outputs.emplace_back(out);
  if (inputs[0].Rank() == 3) outputs.emplace_back(out.AppendDim(inputs[0][2]));
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kUnexpectedError, "Input has a wrong shape");
}
}  // namespace dataset
}  // namespace mindspore
