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

#include "minddata/dataset/text/kernels/data_utils.h"

#include <algorithm>
#include <string>

#include "minddata/dataset/core/pybind_support.h"
#include "minddata/dataset/kernels/data/slice_op.h"
#include "minddata/dataset/kernels/data/concatenate_op.h"

namespace mindspore {
namespace dataset {
Status SlidingWindowHelper(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, TensorShape out_shape,
                           uint32_t width, int32_t axis) {
  // if the data row has fewer items than width, the corresponding result row will be empty
  if (out_shape.Size() == 0) {
    MS_LOG(WARNING) << "The data row has fewer items than width, the result will be empty.";
    return Tensor::CreateEmpty(TensorShape({0}), input->type(), output);
  }

  axis = Tensor::HandleNeg(axis, input->shape().Size());
  int32_t axis_end = input->shape()[axis];
  std::shared_ptr<Tensor> tmp;
  auto concatenate_op = std::make_unique<ConcatenateOp>(axis, nullptr, nullptr);

  // Slice on specified axis and concatenate on new axis
  for (int32_t i = 0; i + width <= axis_end; i++) {
    auto slice_op = std::make_unique<SliceOp>(Slice(i, i + width, 1));
    RETURN_IF_NOT_OK(slice_op->Compute(input, &tmp));
    if (i == 0) {
      *output = tmp;
    } else {
      TensorRow in({*output, tmp});
      TensorRow out_row;
      RETURN_IF_NOT_OK(concatenate_op->Compute(in, &out_row));
      *output = out_row[0];
    }
  }
  RETURN_IF_NOT_OK((*output)->Reshape(out_shape));
  return Status::OK();
}

Status AppendOffsetsHelper(const std::vector<uint32_t> &offsets_start, const std::vector<uint32_t> &offsets_limit,
                           TensorRow *output) {
  std::shared_ptr<Tensor> offsets_start_tensor, offsets_limit_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(offsets_start, &offsets_start_tensor));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(offsets_limit, &offsets_limit_tensor));

  output->push_back(offsets_start_tensor);
  output->push_back(offsets_limit_tensor);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
