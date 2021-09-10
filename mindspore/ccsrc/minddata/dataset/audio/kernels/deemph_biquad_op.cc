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
#include "minddata/dataset/audio/kernels/deemph_biquad_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status DeemphBiquadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  TensorShape input_shape = input->shape();
  CHECK_FAIL_RETURN_UNEXPECTED(input_shape.Size() > 0, "DeemphBiquad: input tensor is not in shape of <..., time>.");
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType(DataType::DE_FLOAT32) ||
                                 input->type() == DataType(DataType::DE_FLOAT16) ||
                                 input->type() == DataType(DataType::DE_FLOAT64),
                               "DeemphBiquad: input tensor type should be float, but got: " + input->type().ToString());
  int32_t central_freq = 0;
  double width_slope = 0.0;
  double gain = 0.0;
  // central_freq, width_slope and gain value reference sox values
  if (sample_rate_ == 44100) {
    central_freq = 5283;
    width_slope = 0.4845;
    gain = -9.477;
  } else if (sample_rate_ == 48000) {
    central_freq = 5356;
    width_slope = 0.479;
    gain = -9.62;
  }

  double w0 = 2 * PI * central_freq / sample_rate_;
  double A = exp(gain / 40 * log(10));
  double alpha = sin(w0) / 2 * sqrt((A + 1 / A) * (1 / width_slope - 1) + 2);

  // temp1, temp2, temp3 are the intermediate variable used to solve for a and b.
  double temp1 = 2 * sqrt(A) * alpha;
  double temp2 = (A - 1) * cos(w0);
  double temp3 = (A + 1) * cos(w0);

  double b0 = A * ((A + 1) + temp2 + temp1);
  double b1 = -2 * A * ((A - 1) + temp3);
  double b2 = A * ((A + 1) + temp2 - temp1);
  double a0 = (A + 1) - temp2 + temp1;
  double a1 = 2 * ((A - 1) - temp3);
  double a2 = (A + 1) - temp2 - temp1;
  if (input->type() == DataType(DataType::DE_FLOAT32)) {
    return Biquad(input, output, static_cast<float>(b0), static_cast<float>(b1), static_cast<float>(b2),
                  static_cast<float>(a0), static_cast<float>(a1), static_cast<float>(a2));
  } else if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Biquad(input, output, static_cast<double>(b0), static_cast<double>(b1), static_cast<double>(b2),
                  static_cast<double>(a0), static_cast<double>(a1), static_cast<double>(a2));
  } else {
    return Biquad(input, output, static_cast<float16>(b0), static_cast<float16>(b1), static_cast<float16>(b2),
                  static_cast<float16>(a0), static_cast<float16>(a1), static_cast<float16>(a2));
  }
}
}  // namespace dataset
}  // namespace mindspore
