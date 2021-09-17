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
#include "minddata/dataset/audio/kernels/vol_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status VolOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  std::shared_ptr<Tensor> input_tensor;
  TensorShape input_shape = input->shape();
  CHECK_FAIL_RETURN_UNEXPECTED(input_shape.Size() > 0, "Vol: input tensor is not in shape of <..., time>.");
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->type().IsNumeric(),
    "Vol: input tensor type should be int, float or double, but got: " + input->type().ToString());
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return Vol(input_tensor, output, gain_, gain_type_);
  } else {
    input_tensor = input;
    return Vol(input_tensor, output, static_cast<double>(gain_), gain_type_);
  }
}

Status VolOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  if (!inputs[0].IsNumeric()) {
    RETURN_STATUS_UNEXPECTED("Vol: input tensor type should be int, float or double, but got: " + inputs[0].ToString());
  } else if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
