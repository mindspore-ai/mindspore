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
#include <utility>

#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/kernels/image/bounding_box.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/random_crop_and_resize_with_bbox_op.h"

namespace mindspore {
namespace dataset {

Status RandomCropAndResizeWithBBoxOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  RETURN_IF_NOT_OK(BoundingBox::ValidateBoundingBoxes(input));
  CHECK_FAIL_RETURN_UNEXPECTED(input[0]->shape().Size() >= 2,
                               "RandomCropAndResizeWithBBox: image shape is not <H,W,C> or <H,W>.");

  output->resize(2);
  (*output)[1] = std::move(input[1]);  // move boxes over to output

  size_t bboxCount = input[1]->shape()[0];  // number of rows in bbox tensor
  int h_in = input[0]->shape()[0];
  int w_in = input[0]->shape()[1];
  int x = 0;
  int y = 0;
  int crop_height = 0;
  int crop_width = 0;

  RETURN_IF_NOT_OK(RandomCropAndResizeOp::GetCropBox(h_in, w_in, &x, &y, &crop_height, &crop_width));

  int maxX = x + crop_width;  // max dims of selected CropBox on image
  int maxY = y + crop_height;

  RETURN_IF_NOT_OK(BoundingBox::UpdateBBoxesForCrop(&(*output)[1], &bboxCount, x, y, maxX, maxY));  // IMAGE_UTIL
  RETURN_IF_NOT_OK(CropAndResize(input[0], &(*output)[0], x, y, crop_height, crop_width, target_height_, target_width_,
                                 interpolation_));

  RETURN_IF_NOT_OK(BoundingBox::UpdateBBoxesForResize((*output)[1], bboxCount, target_width_, target_height_,
                                                      crop_width, crop_height));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
