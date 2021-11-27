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
#include "minddata/dataset/audio/kernels/phaser_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"

namespace mindspore {
namespace dataset {
PhaserOp::PhaserOp(int32_t sample_rate, float gain_in, float gain_out, float delay_ms, float decay, float mod_speed,
                   bool sinusoidal)
    : sample_rate_(sample_rate),
      gain_in_(gain_in),
      gain_out_(gain_out),
      delay_ms_(delay_ms),
      decay_(decay),
      mod_speed_(mod_speed),
      sinusoidal_(sinusoidal) {}

Status PhaserOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input tensor dimension, it should be greater than 0.
  RETURN_IF_NOT_OK(ValidateLowRank("Phaser", input, kMinAudioDim, "<..., time>"));
  // check input type, it should be DE_FLOAT
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Phaser", input));
  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return Phaser<float>(input_tensor, output, sample_rate_, gain_in_, gain_out_, delay_ms_, decay_, mod_speed_,
                         sinusoidal_);
  } else {
    input_tensor = input;
    return Phaser<double>(input_tensor, output, sample_rate_, gain_in_, gain_out_, delay_ms_, decay_, mod_speed_,
                          sinusoidal_);
  }
}

Status PhaserOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("Phaser", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
