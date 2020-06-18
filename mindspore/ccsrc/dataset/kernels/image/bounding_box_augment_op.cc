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
#include "dataset/kernels/image/bounding_box_augment_op.h"
#include "dataset/kernels/image/resize_op.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/core/cv_tensor.h"

namespace mindspore {
namespace dataset {
const float BoundingBoxAugOp::defRatio = 0.3;

BoundingBoxAugOp::BoundingBoxAugOp(std::shared_ptr<TensorOp> transform, float ratio)
    : ratio_(ratio), transform_(std::move(transform)) {}

Status BoundingBoxAugOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  BOUNDING_BOX_CHECK(input);  // check if bounding boxes are valid
  uint32_t num_of_boxes = input[1]->shape()[0];
  uint32_t num_to_aug = num_of_boxes * ratio_;  // cast to int
  std::vector<uint32_t> boxes(num_of_boxes);
  std::vector<uint32_t> selected_boxes;
  for (uint32_t i = 0; i < num_of_boxes; i++) boxes[i] = i;
  // sample bboxes according to ratio picked by user
  std::random_device rd;
  std::sample(boxes.begin(), boxes.end(), std::back_inserter(selected_boxes), num_to_aug, std::mt19937(rd()));
  std::shared_ptr<Tensor> crop_out;
  std::shared_ptr<Tensor> res_out;
  std::shared_ptr<CVTensor> input_restore = CVTensor::AsCVTensor(input[0]);

  for (uint32_t i = 0; i < num_to_aug; i++) {
    uint32_t min_x = 0;
    uint32_t min_y = 0;
    uint32_t b_w = 0;
    uint32_t b_h = 0;
    // get the required items
    input[1]->GetItemAt<uint32_t>(&min_x, {selected_boxes[i], 0});
    input[1]->GetItemAt<uint32_t>(&min_y, {selected_boxes[i], 1});
    input[1]->GetItemAt<uint32_t>(&b_w, {selected_boxes[i], 2});
    input[1]->GetItemAt<uint32_t>(&b_h, {selected_boxes[i], 3});
    Crop(input_restore, &crop_out, min_x, min_y, b_w, b_h);
    // transform the cropped bbox region
    transform_->Compute(crop_out, &res_out);
    // place the transformed region back in the restored input
    std::shared_ptr<CVTensor> res_img = CVTensor::AsCVTensor(res_out);
    // check if transformed crop is out of bounds of the box
    if (res_img->mat().cols > b_w || res_img->mat().rows > b_h || res_img->mat().cols < b_w ||
        res_img->mat().rows < b_h) {
      // if so, resize to fit in the box
      std::shared_ptr<TensorOp> resize_op = std::make_shared<ResizeOp>(b_h, b_w);
      resize_op->Compute(std::static_pointer_cast<Tensor>(res_img), &res_out);
      res_img = CVTensor::AsCVTensor(res_out);
    }
    res_img->mat().copyTo(input_restore->mat()(cv::Rect(min_x, min_y, res_img->mat().cols, res_img->mat().rows)));
  }
  (*output).push_back(std::move(std::static_pointer_cast<Tensor>(input_restore)));
  (*output).push_back(input[1]);
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
