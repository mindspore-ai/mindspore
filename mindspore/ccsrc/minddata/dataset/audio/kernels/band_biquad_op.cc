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
#include "minddata/dataset/audio/kernels/band_biquad_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status BandBiquadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  TensorShape input_shape = input->shape();
  // check input tensor dimension, it should be greater than 0.
  CHECK_FAIL_RETURN_UNEXPECTED(input_shape.Size() > 0, "BandBiquad: input tensor is not in shape of <..., time>.");
  // check input type, it should be DE_FLOAT32 or DE_FLOAT16 or DE_FLOAT64
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType(DataType::DE_FLOAT32) ||
                                 input->type() == DataType(DataType::DE_FLOAT16) ||
                                 input->type() == DataType(DataType::DE_FLOAT64),
                               "BandBiquad: input tensor type should be float, but got: " + input->type().ToString());
  double w0 = 2 * PI * central_freq_ / sample_rate_;
  double bw_Hz = central_freq_ / Q_;
  double a0 = 1.;
  double a2 = exp(-2 * PI * bw_Hz / sample_rate_);
  double a1 = -4 * a2 / (1 + a2) * cos(w0);
  CHECK_FAIL_RETURN_UNEXPECTED(a2 != 0, "BandBiquad: ZeroDivisionError.");
  double b0 = sqrt(1 - a1 * a1 / (4 * a2)) * (1 - a2);
  if (noise_) {
    CHECK_FAIL_RETURN_UNEXPECTED(b0 != 0, "BandBiquad: ZeroDivisionError.");
    double mutl = sqrt(((1 + a2) * (1 + a2) - a1 * a1) * (1 - a2) / (1 + a2)) / b0;
    b0 *= mutl;
  }
  double b1 = 0.;
  double b2 = 0.;
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
