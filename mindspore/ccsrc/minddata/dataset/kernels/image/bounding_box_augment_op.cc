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

#include <vector>
#include <utility>
#include "minddata/dataset/kernels/image/bounding_box_augment_op.h"
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
  BOUNDING_BOX_CHECK(input);  // check if bounding boxes are valid
  uint32_t num_of_boxes = input[1]->shape()[0];
  std::shared_ptr<Tensor> crop_out;
  std::shared_ptr<Tensor> res_out;
  std::shared_ptr<CVTensor> input_restore = CVTensor::AsCVTensor(input[0]);
  for (uint32_t i = 0; i < num_of_boxes; i++) {
    // using a uniform distribution to ensure op happens with probability ratio_
    if (uniform_(rnd_) < ratio_) {
      float min_x = 0;
      float min_y = 0;
      float b_w = 0;
      float b_h = 0;
      // get the required items
      RETURN_IF_NOT_OK(input[1]->GetItemAt<float>(&min_x, {i, 0}));
      RETURN_IF_NOT_OK(input[1]->GetItemAt<float>(&min_y, {i, 1}));
      RETURN_IF_NOT_OK(input[1]->GetItemAt<float>(&b_w, {i, 2}));
      RETURN_IF_NOT_OK(input[1]->GetItemAt<float>(&b_h, {i, 3}));
      RETURN_IF_NOT_OK(Crop(input_restore, &crop_out, static_cast<int>(min_x), static_cast<int>(min_y),
                            static_cast<int>(b_w), static_cast<int>(b_h)));
      // transform the cropped bbox region
      RETURN_IF_NOT_OK(transform_->Compute(crop_out, &res_out));
      // place the transformed region back in the restored input
      std::shared_ptr<CVTensor> res_img = CVTensor::AsCVTensor(res_out);
      // check if transformed crop is out of bounds of the box
      if (res_img->mat().cols > b_w || res_img->mat().rows > b_h || res_img->mat().cols < b_w ||
          res_img->mat().rows < b_h) {
        // if so, resize to fit in the box
        std::shared_ptr<TensorOp> resize_op =
          std::make_shared<ResizeOp>(static_cast<int32_t>(b_h), static_cast<int32_t>(b_w));
        RETURN_IF_NOT_OK(resize_op->Compute(std::static_pointer_cast<Tensor>(res_img), &res_out));
        res_img = CVTensor::AsCVTensor(res_out);
      }
      res_img->mat().copyTo(input_restore->mat()(cv::Rect(min_x, min_y, res_img->mat().cols, res_img->mat().rows)));
    }
  }
  (*output).push_back(std::move(std::static_pointer_cast<Tensor>(input_restore)));
  (*output).push_back(input[1]);
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
