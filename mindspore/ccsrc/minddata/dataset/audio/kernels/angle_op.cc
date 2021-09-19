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
#include <cmath>

#include "minddata/dataset/audio/kernels/angle_op.h"
#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"

namespace mindspore {
namespace dataset {
Status AngleOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // if If the last dimension is not 2, then it's not a complex number
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape()[-1] == 2, "Angle: input tensor is not in shape of <..., complex=2>.");
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->type().IsNumeric(),
    "Angle: input tensor type should be int, float or double, but got: " + input->type().ToString());
  if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Angle<double>(input, output);
  } else {
    std::shared_ptr<Tensor> tmp;
    TypeCast(input, &tmp, DataType(DataType::DE_FLOAT32));
    return Angle<float>(tmp, output);
  }
}

Status AngleOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  std::vector shape = inputs[0].AsVector();

  shape.pop_back();
  TensorShape out = TensorShape{shape};
  outputs.emplace_back(out);
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kMDUnexpectedError, "Angle: invalid input wrong shape.");
}

Status AngleOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
