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

#include "dataset/util/status.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/kernels/image/random_vertical_flip_with_bbox_op.h"

namespace mindspore {
namespace dataset {
const float RandomVerticalFlipWithBBoxOp::kDefProbability = 0.5;
Status RandomVerticalFlipWithBBoxOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  BOUNDING_BOX_CHECK(input);

  if (distribution_(rnd_)) {
    dsize_t imHeight = input[0]->shape()[0];
    size_t boxCount = input[1]->shape()[0];  // number of rows in tensor

    // one time allocation -> updated in the loop
    // type defined based on VOC test dataset
    for (int i = 0; i < boxCount; i++) {
      uint32_t boxCorner_y = 0;
      uint32_t boxHeight = 0;
      uint32_t newBoxCorner_y = 0;
      RETURN_IF_NOT_OK(input[1]->GetUnsignedIntAt(&boxCorner_y, {i, 1}));  // get min y of bbox
      RETURN_IF_NOT_OK(input[1]->GetUnsignedIntAt(&boxHeight, {i, 3}));    // get height of bbox

      // subtract (curCorner + height) from (max) for new Corner position
      newBoxCorner_y = (imHeight - 1) - ((boxCorner_y + boxHeight) - 1);
      RETURN_IF_NOT_OK(input[1]->SetItemAt({i, 1}, newBoxCorner_y));
    }

    (*output).push_back(nullptr);
    (*output).push_back(nullptr);
    (*output)[1] = std::move(input[1]);

    return VerticalFlip(input[0], &(*output)[0]);
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
