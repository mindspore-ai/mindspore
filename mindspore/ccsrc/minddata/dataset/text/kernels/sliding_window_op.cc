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
#include "minddata/dataset/text/kernels/sliding_window_op.h"

namespace mindspore {
namespace dataset {
Status SlidingWindowOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Rank() == 1,
                               "SlidingWindow: SlidingWindow supports 1D input only for now.");
  CHECK_FAIL_RETURN_UNEXPECTED(axis_ == 0 || axis_ == -1, "SlidingWindow: axis supports 0 or -1 only for now.");

  std::vector<TensorShape> input_shape = {input->shape()};
  std::vector<TensorShape> output_shape = {TensorShape({})};
  RETURN_IF_NOT_OK(OutputShape(input_shape, output_shape));

  RETURN_IF_NOT_OK(SlidingWindowHelper(input, output, output_shape[0], width_, axis_));
  return Status::OK();
}

Status SlidingWindowOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  CHECK_FAIL_RETURN_UNEXPECTED(inputs.size() == NumInput(), "SlidingWindow: incorrect number of inputs\n");
  int32_t axis = Tensor::HandleNeg(axis_, inputs[0].Size());
  TensorShape input_shape = inputs[0];
  std::vector<dsize_t> output_shape_initializer;

  // if a data row has fewer items than width, the corresponding result row will be empty.
  if (input_shape[axis] >= width_) {
    for (int32_t idx = 0; idx < input_shape.Size(); ++idx) {
      if (idx != axis) {
        output_shape_initializer.push_back(input_shape[idx]);
      } else {
        output_shape_initializer.push_back(input_shape[idx] - (width_ - 1));
        output_shape_initializer.push_back(width_);
      }
    }
  }

  outputs.pop_back();
  outputs.emplace_back(TensorShape(output_shape_initializer));
  CHECK_FAIL_RETURN_UNEXPECTED(outputs.size() == NumOutput(), "SlidingWindow: incorrect number of outputs\n");
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
