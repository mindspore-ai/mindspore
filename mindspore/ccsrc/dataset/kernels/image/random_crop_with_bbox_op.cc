/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <utility>

#include "dataset/kernels/image/random_crop_with_bbox_op.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/util/random.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status RandomCropWithBBoxOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  BOUNDING_BOX_CHECK(input);

  std::shared_ptr<Tensor> pad_image;
  int32_t t_pad_top, t_pad_bottom, t_pad_left, t_pad_right;
  size_t boxCount = input[1]->shape()[0];  // number of rows

  int32_t padded_image_h;
  int32_t padded_image_w;

  output->resize(2);
  (*output)[1] = std::move(input[1]);  // since some boxes may be removed

  bool crop_further = true;  // Whether further cropping will be required or not, true unless required size matches
  RETURN_IF_NOT_OK(          // Error passed back to caller
    RandomCropOp::ImagePadding(input[0], &pad_image, &t_pad_top, &t_pad_bottom, &t_pad_left, &t_pad_right,
                               &padded_image_w, &padded_image_h, &crop_further));

  // update bounding boxes with new values based on relevant image padding
  if (t_pad_left || t_pad_bottom) {
    RETURN_IF_NOT_OK(PadBBoxes(&(*output)[1], boxCount, t_pad_left, t_pad_top));
  }
  if (!crop_further) {
    // no further cropping required
    (*output)[0] = pad_image;
    (*output)[1] = std::move(input[1]);
    return Status::OK();
  }

  int x, y;
  RandomCropOp::GenRandomXY(&x, &y, padded_image_w, padded_image_h);
  int maxX = x + RandomCropOp::crop_width_;  // max dims of selected CropBox on image
  int maxY = y + RandomCropOp::crop_height_;
  RETURN_IF_NOT_OK(UpdateBBoxesForCrop(&(*output)[1], &boxCount, x, y, maxX, maxY));
  return Crop(pad_image, &(*output)[0], x, y, RandomCropOp::crop_width_, RandomCropOp::crop_height_);
}
}  // namespace dataset
}  // namespace mindspore
