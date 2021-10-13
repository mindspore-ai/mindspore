/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/audio/kernels/overdrive_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"

namespace mindspore {
namespace dataset {
OverdriveOp::OverdriveOp(float gain, float color) : gain_(gain), color_(color) {}

Status OverdriveOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  TensorShape input_shape = input->shape();
  // check input tensor dimension, it should be greater than 0.
  CHECK_FAIL_RETURN_UNEXPECTED(input_shape.Size() > 0, "Overdrive: input tensor is not in shape of <..., time>.");
  // check input type, it should be numeric.
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->type().IsNumeric(),
    "Overdrive: input tensor type should be int, float or double, but got: " + input->type().ToString());
  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return Overdrive<float>(input_tensor, output, gain_, color_);
  } else {
    input_tensor = input;
    return Overdrive<double>(input_tensor, output, gain_, color_);
  }
}

Status OverdriveOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  if (!inputs[0].IsNumeric()) {
    RETURN_STATUS_UNEXPECTED("Overdrive: input tensor type should be int, float or double, but got: " +
                             inputs[0].ToString());
  } else if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
