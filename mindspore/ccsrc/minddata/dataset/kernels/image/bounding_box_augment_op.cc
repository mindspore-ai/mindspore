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

#include "minddata/dataset/kernels/image/bounding_box_augment_op.h"
#include "minddata/dataset/kernels/image/bounding_box.h"
#include "minddata/dataset/kernels/image/resize_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/core/cv_tensor.h"

namespace mindspore {
namespace dataset {
const float BoundingBoxAugmentOp::kDefRatio = 0.3;

BoundingBoxAugmentOp::BoundingBoxAugmentOp(std::shared_ptr<TensorOp> transform, float ratio)
    : ratio_(ratio), uniform_(0, 1), transform_(std::move(transform)) {
  rnd_.seed(GetSeed());
}

Status BoundingBoxAugmentOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  RETURN_IF_NOT_OK(BoundingBox::ValidateBoundingBoxes(input));
  uint32_t num_of_boxes = input[1]->shape()[0];
  std::shared_ptr<Tensor> crop_out;
  std::shared_ptr<Tensor> res_out;
  std::shared_ptr<CVTensor> input_restore = CVTensor::AsCVTensor(input[0]);
  for (uint32_t i = 0; i < num_of_boxes; i++) {
    // using a uniform distribution to ensure op happens with probability ratio_
    if (uniform_(rnd_) < ratio_) {
      std::shared_ptr<BoundingBox> bbox;
      RETURN_IF_NOT_OK(BoundingBox::ReadFromTensor(input[1], i, &bbox));
      RETURN_IF_NOT_OK(Crop(input_restore, &crop_out, static_cast<int>(bbox->x()), static_cast<int>(bbox->y()),
                            static_cast<int>(bbox->width()), static_cast<int>(bbox->height())));
      // transform the cropped bbox region
      TensorRow crop_out_row;
      TensorRow res_out_row;
      crop_out_row.push_back(crop_out);
      res_out_row.push_back(res_out);
      RETURN_IF_NOT_OK(transform_->Compute(crop_out_row, &res_out_row));
      // place the transformed region back in the restored input
      std::shared_ptr<CVTensor> res_img = CVTensor::AsCVTensor(res_out_row[0]);
      // check if transformed crop is out of bounds of the box
      if (res_img->mat().cols > bbox->width() || res_img->mat().rows > bbox->height() ||
          res_img->mat().cols < bbox->width() || res_img->mat().rows < bbox->height()) {
        // if so, resize to fit in the box
        std::shared_ptr<TensorOp> resize_op =
          std::make_shared<ResizeOp>(static_cast<int32_t>(bbox->height()), static_cast<int32_t>(bbox->width()));
        RETURN_IF_NOT_OK(resize_op->Compute(std::static_pointer_cast<Tensor>(res_img), &res_out_row[0]));
        res_img = CVTensor::AsCVTensor(res_out_row[0]);
      }
      res_img->mat().copyTo(
        input_restore->mat()(cv::Rect(bbox->x(), bbox->y(), res_img->mat().cols, res_img->mat().rows)));
    }
  }
  (*output).push_back(std::move(std::static_pointer_cast<Tensor>(input_restore)));
  (*output).push_back(input[1]);
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
