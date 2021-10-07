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

#include "minddata/dataset/util/status.h"
#include "minddata/dataset/kernels/image/bounding_box.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/random_vertical_flip_with_bbox_op.h"

namespace mindspore {
namespace dataset {
const float RandomVerticalFlipWithBBoxOp::kDefProbability = 0.5;
Status RandomVerticalFlipWithBBoxOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  RETURN_IF_NOT_OK(BoundingBox::ValidateBoundingBoxes(input));

  if (distribution_(rnd_)) {
    dsize_t imHeight = input[0]->shape()[0];
    size_t boxCount = input[1]->shape()[0];  // number of rows in tensor

    // one time allocation -> updated in the loop
    // type defined based on VOC test dataset
    for (int i = 0; i < boxCount; i++) {
      std::shared_ptr<BoundingBox> bbox;
      RETURN_IF_NOT_OK(BoundingBox::ReadFromTensor(input[1], i, &bbox));

      // subtract (curCorner + height) from (max) for new Corner position
      BoundingBox::bbox_float newBoxCorner_y = (imHeight - 1.0) - ((bbox->y() + bbox->height()) - 1.0);
      bbox->SetY(newBoxCorner_y);
      RETURN_IF_NOT_OK(bbox->WriteToTensor(input[1], i));
    }
    const int output_count = 2;
    output->resize(output_count);
    (*output)[1] = input[1];

    return VerticalFlip(input[0], &(*output)[0]);
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
