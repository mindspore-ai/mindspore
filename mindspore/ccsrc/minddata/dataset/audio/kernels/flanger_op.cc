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
#include "minddata/dataset/audio/kernels/flanger_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status FlangerOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input dimensions, it should be 2 dimensions or more
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Rank() >= 2,
                               "Flanger: input tensor is not in shape of <..., channel, time>.");

  // check input channel, it should be less than or equal to 4
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape()[-2] <= 4,
                               "Flanger: the channel of input tensor must be less than or equal to 4, but got: " +
                                 std::to_string(input->shape()[-2]));

  // check input type, it should be [int, float, double]
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->type().IsNumeric(),
    "Flanger: input tensor type should be int, float or double, but got: " + input->type().ToString());

  if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Flanger<double>(input, output, sample_rate_, delay_, depth_, regen_, width_, speed_, phase_, Modulation_,
                           Interpolation_);
  } else {
    return Flanger<float>(input, output, sample_rate_, delay_, depth_, regen_, width_, speed_, phase_, Modulation_,
                          Interpolation_);
  }
}

Status FlangerOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  CHECK_FAIL_RETURN_UNEXPECTED(
    inputs[0].IsNumeric(),
    "Flanger: input tensor type should be int, float or double, but got: " + inputs[0].ToString());
  outputs[0] = inputs[0];
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
