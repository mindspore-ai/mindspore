/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/resize_with_bbox_op.h"

#include <memory>

#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/bounding_box.h"
#include "minddata/dataset/kernels/image/resize_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status ResizeWithBBoxOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  RETURN_IF_NOT_OK(BoundingBox::ValidateBoundingBoxes(input));

  auto input_h = static_cast<int32_t>(input[0]->shape()[0]);
  auto input_w = static_cast<int32_t>(input[0]->shape()[1]);

  output->resize(2);
  (*output)[1] = input[1];  // move boxes over to output

  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input[0]);

  RETURN_IF_NOT_OK(ResizeOp::Compute(std::static_pointer_cast<Tensor>(input_cv), &(*output)[0]));

  auto output_h = static_cast<int32_t>((*output)[0]->shape()[0]);  // output height if ResizeWithBBox
  auto output_w = static_cast<int32_t>((*output)[0]->shape()[1]);  // output width if ResizeWithBBox

  size_t bboxCount = input[1]->shape()[0];  // number of rows in bbox tensor
  RETURN_IF_NOT_OK(BoundingBox::UpdateBBoxesForResize((*output)[1], bboxCount, output_w, output_h, input_w, input_h));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
