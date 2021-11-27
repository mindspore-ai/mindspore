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

#include "minddata/dataset/audio/kernels/db_to_amplitude_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status DBToAmplitudeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input tensor dimension, it should be greater than 0.
  RETURN_IF_NOT_OK(ValidateLowRank("DBToAmplitude", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("DBToAmplitude", input));

  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType(DataType::DE_FLOAT64)) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return DBToAmplitude<float>(input_tensor, output, ref_, power_);
  } else {
    input_tensor = input;
    return DBToAmplitude<double>(input_tensor, output, ref_, power_);
  }
}
}  // namespace dataset
}  // namespace mindspore
