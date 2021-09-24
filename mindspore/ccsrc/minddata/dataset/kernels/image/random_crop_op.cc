/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/random_crop_op.h"
#include <random>
#include <tuple>
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

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
                           int32_t pad_left, int32_t pad_right, bool pad_if_needed, BorderType padding_mode,
                           uint8_t fill_r, uint8_t fill_g, uint8_t fill_b)
    : crop_height_(crop_height),
      crop_width_(crop_width),
      pad_top_(pad_top),
      pad_bottom_(pad_bottom),
      pad_left_(pad_left),
      pad_right_(pad_right),
      pad_if_needed_(pad_if_needed),
      border_type_(padding_mode),
      fill_r_(fill_r),
      fill_g_(fill_g),
      fill_b_(fill_b) {
  rnd_.seed(GetSeed());
  is_deterministic_ = false;
}

Status RandomCropOp::ImagePadding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *pad_image,
                                  int32_t *t_pad_top, int32_t *t_pad_bottom, int32_t *t_pad_left, int32_t *t_pad_right,
                                  int32_t *padded_image_w, int32_t *padded_image_h, bool *crop_further) {
  *t_pad_top = pad_top_;
  *t_pad_bottom = pad_bottom_;
  *t_pad_left = pad_left_;
  *t_pad_right = pad_right_;

  constexpr int64_t max_ratio = 3;
  CHECK_FAIL_RETURN_UNEXPECTED(
    pad_top_ < input->shape()[0] * max_ratio && pad_bottom_ < input->shape()[0] * max_ratio &&
      pad_left_ < input->shape()[1] * max_ratio && pad_right_ < input->shape()[1] * max_ratio,
    "Pad: padding size is three times bigger than the image size, padding top: " + std::to_string(pad_top_) +
      ", padding bottom: " + std::to_string(pad_bottom_) + ", padding pad_left_: " + std::to_string(pad_left_) +
      ", padding padding right:" + std::to_string(pad_right_) + ", image shape: " + std::to_string(input->shape()[0]) +
      ", " + std::to_string(input->shape()[1]));

  RETURN_IF_NOT_OK(
    Pad(input, pad_image, pad_top_, pad_bottom_, pad_left_, pad_right_, border_type_, fill_r_, fill_g_, fill_b_));
  CHECK_FAIL_RETURN_UNEXPECTED(
    (*pad_image)->shape().Size() >= 2,
    "RandomCrop: invalid shape of image after pad, got rank: " + std::to_string((*pad_image)->shape().Size()));

  *padded_image_h = (*pad_image)->shape()[0];
  *padded_image_w = (*pad_image)->shape()[1];

  if (*padded_image_h == crop_height_ && *padded_image_w == crop_width_) {
    *crop_further = false;  //  no need for further crop
    return Status::OK();
  } else if (pad_if_needed_) {
    // check the dimensions of the image for padding, if we do need padding, then we change the pad values
    if (*padded_image_h < crop_height_) {
      RETURN_IF_NOT_OK(Pad(*pad_image, pad_image, crop_height_ - *padded_image_h, crop_height_ - *padded_image_h, 0, 0,
                           border_type_, fill_r_, fill_g_, fill_b_));

      // update pad total above/below
      t_pad_top += (crop_height_ - *padded_image_h);
      t_pad_bottom += (crop_height_ - *padded_image_h);
    }
    if (*padded_image_w < crop_width_) {
      RETURN_IF_NOT_OK(Pad(*pad_image, pad_image, 0, 0, crop_width_ - *padded_image_w, crop_width_ - *padded_image_w,
                           border_type_, fill_r_, fill_g_, fill_b_));
      // update pad total left/right
      t_pad_left += (crop_width_ - *padded_image_w);
      t_pad_right += (crop_width_ - *padded_image_w);
    }
    *padded_image_h = (*pad_image)->shape()[0];
    *padded_image_w = (*pad_image)->shape()[1];
  }

  if (crop_height_ == 0 || crop_width_ == 0) {
    return Status(StatusCode::kMDShapeMisMatch, __LINE__, __FILE__,
                  "RandomCrop: invalid crop size, crop width or crop height is not allowed to be zero.");
  }
  if (*padded_image_h < crop_height_ || *padded_image_w < crop_width_ || crop_height_ == 0 || crop_width_ == 0) {
    return Status(StatusCode::kMDShapeMisMatch, __LINE__, __FILE__,
                  "RandomCrop: invalid crop size, crop size is bigger than the image dimensions, got crop height: " +
                    std::to_string(crop_height_) + ", crop width: " + std::to_string(crop_width_));
  }
  return Status::OK();
}

void RandomCropOp::GenRandomXY(int *x, int *y, const int32_t &padded_image_w, const int32_t &padded_image_h) {
  // GenCropPoints for cropping
  *x = std::uniform_int_distribution<int>(0, padded_image_w - crop_width_)(rnd_);
  *y = std::uniform_int_distribution<int>(0, padded_image_h - crop_height_)(rnd_);
}

Status RandomCropOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  if (input.size() != 1) {
    for (size_t i = 0; i < input.size() - 1; i++) {
      if (input[i]->Rank() != 2 && input[i]->Rank() != 3) {
        std::string err_msg =
          "RandomCropOp: image shape is not <H,W,C> or <H, W>, but got rank:" + std::to_string(input[i]->Rank());
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      if (input[i]->shape()[0] != input[i + 1]->shape()[0] || input[i]->shape()[1] != input[i + 1]->shape()[1]) {
        RETURN_STATUS_UNEXPECTED("RandomCropOp: Input images must have the same size.");
      }
    }
  }
  int x = 0;
  int y = 0;
  const int output_count = input.size();
  output->resize(output_count);
  for (size_t i = 0; i < input.size(); i++) {
    RETURN_IF_NOT_OK(ValidateImageRank("RandomCrop", input[i]->shape().Size()));
    std::shared_ptr<Tensor> pad_image = nullptr;
    int32_t t_pad_top = 0;
    int32_t t_pad_bottom = 0;
    int32_t t_pad_left = 0;
    int32_t t_pad_right = 0;
    int32_t padded_image_w = 0;
    int32_t padded_image_h = 0;
    bool crop_further = true;  // whether image needs further cropping based on new size & requirements

    RETURN_IF_NOT_OK(  // error code sent back directly
      ImagePadding(input[i], &pad_image, &t_pad_top, &t_pad_bottom, &t_pad_left, &t_pad_right, &padded_image_w,
                   &padded_image_h, &crop_further));
    if (!crop_further) {
      (*output)[i] = pad_image;
      continue;
    }
    if (i == 0) {
      GenRandomXY(&x, &y, padded_image_w, padded_image_h);
    }
    RETURN_IF_NOT_OK(Crop(pad_image, &(*output)[i], x, y, crop_width_, crop_height_));
  }
  return Status::OK();
}

Status RandomCropOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out = TensorShape{crop_height_, crop_width_};
  if (inputs[0].Rank() == 2) {
    (void)outputs.emplace_back(out);
  }
  if (inputs[0].Rank() == 3) {
    (void)outputs.emplace_back(out.AppendDim(inputs[0][2]));
  }
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError,
                "RandomCrop: invalid input shape, expected 2D or 3D input, but got input dimension is:" +
                  std::to_string(inputs[0].Rank()));
}
}  // namespace dataset
}  // namespace mindspore
