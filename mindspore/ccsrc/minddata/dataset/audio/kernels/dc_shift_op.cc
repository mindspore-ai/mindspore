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
#include "minddata/dataset/audio/kernels/dc_shift_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"

namespace mindspore {
namespace dataset {
Status DCShiftOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // input <..., time>.
  CHECK_FAIL_RETURN_UNEXPECTED(input->Rank() > 0, "ComplexNorm: input tensor is not in shape of <..., time>.");
  // If datatype is not a numeric type, then we cannot deal with the data.
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->type().IsNumeric(),
    "DCShift: input tensor type should be int, float or double, but got: " + input->type().ToString());
  if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return DCShift<double>(input, output, shift_, limiter_gain_);
  } else {
    std::shared_ptr<Tensor> tmp;
    RETURN_IF_NOT_OK(TypeCast(input, &tmp, DataType(DataType::DE_FLOAT32)));
    return DCShift<float>(tmp, output, shift_, limiter_gain_);
  }
}

Status DCShiftOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  CHECK_FAIL_RETURN_UNEXPECTED(
    inputs[0].IsNumeric(),
    "DCShift: input tensor type should be int, float or double, but got: " + inputs[0].ToString());
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
