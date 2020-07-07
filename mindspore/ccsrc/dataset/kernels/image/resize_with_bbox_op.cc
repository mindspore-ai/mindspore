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

#include "dataset/kernels/image/resize_with_bbox_op.h"
#include <utility>
#include <memory>
#include "dataset/kernels/image/resize_op.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/core/pybind_support.h"
#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {

Status ResizeWithBBoxOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  BOUNDING_BOX_CHECK(input);

  int32_t input_h = input[0]->shape()[0];
  int32_t input_w = input[0]->shape()[1];

  output->resize(2);
  (*output)[1] = std::move(input[1]);  // move boxes over to output

  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input[0]));

  RETURN_IF_NOT_OK(ResizeOp::Compute(std::static_pointer_cast<Tensor>(input_cv), &(*output)[0]));

  int32_t output_h = (*output)[0]->shape()[0];  // output height if ResizeWithBBox
  int32_t output_w = (*output)[0]->shape()[1];  // output width if ResizeWithBBox

  size_t bboxCount = input[1]->shape()[0];  // number of rows in bbox tensor
  RETURN_IF_NOT_OK(UpdateBBoxesForResize((*output)[1], bboxCount, output_w, output_h, input_w, input_h));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
