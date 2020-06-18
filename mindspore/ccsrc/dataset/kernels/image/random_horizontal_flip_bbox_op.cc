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
#include <utility>
#include "dataset/kernels/image/random_horizontal_flip_bbox_op.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/util/status.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/core/pybind_support.h"

namespace mindspore {
namespace dataset {
const float RandomHorizontalFlipWithBBoxOp::kDefProbability = 0.5;

Status RandomHorizontalFlipWithBBoxOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  BOUNDING_BOX_CHECK(input);
  if (distribution_(rnd_)) {
    // To test bounding boxes algorithm, create random bboxes from image dims
    size_t numOfBBoxes = input[1]->shape()[0];     // set to give number of bboxes
    float imgCenter = (input[0]->shape()[1] / 2);  // get the center of the image

    for (int i = 0; i < numOfBBoxes; i++) {
      uint32_t b_w = 0;  // bounding box width
      uint32_t min_x = 0;
      // get the required items
      input[1]->GetItemAt<uint32_t>(&min_x, {i, 0});
      input[1]->GetItemAt<uint32_t>(&b_w, {i, 2});
      // do the flip
      float diff = imgCenter - min_x;          // get distance from min_x to center
      uint32_t refl_min_x = diff + imgCenter;  // get reflection of min_x
      uint32_t new_min_x = refl_min_x - b_w;   // subtract from the reflected min_x to get the new one

      input[1]->SetItemAt<uint32_t>({i, 0}, new_min_x);
    }
    (*output).push_back(nullptr);
    (*output).push_back(nullptr);
    // move input to output pointer of bounding boxes
    (*output)[1] = std::move(input[1]);
    // perform HorizontalFlip on the image
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input[0]));
    return HorizontalFlip(std::static_pointer_cast<Tensor>(input_cv), &(*output)[0]);
  }
  *output = input;
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
