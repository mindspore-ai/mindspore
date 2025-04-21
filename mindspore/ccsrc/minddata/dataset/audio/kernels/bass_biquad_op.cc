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
#include "minddata/dataset/audio/kernels/bass_biquad_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status BassBiquadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("BassBiquad", input, kMinAudioDim, "<..., time>"));
  // check input type, it should be DE_FLOAT32 or DE_FLOAT16 or DE_FLOAT64
  RETURN_IF_NOT_OK(ValidateTensorFloat("BassBiquad", input));
  double w0 = 2 * PI * central_freq_ / sample_rate_;
  double alpha = sin(w0) / 2 / Q_;
  double A = exp(gain_ / 40 * log(10));

  double temp1 = 2 * sqrt(A) * alpha;
  double temp2 = (A - 1) * cos(w0);
  double temp3 = (A + 1) * cos(w0);

  double b0 = A * ((A + 1) - temp2 + temp1);
  double b1 = 2 * A * ((A - 1) - temp3);
  double b2 = A * ((A + 1) - temp2 - temp1);
  double a0 = (A + 1) + temp2 + temp1;
  double a1 = -2 * ((A - 1) + temp3);
  double a2 = (A + 1) + temp2 - temp1;
  if (input->type() == DataType(DataType::DE_FLOAT32)) {
    return Biquad(input, output, static_cast<float>(b0 / a0), static_cast<float>(b1 / a0), static_cast<float>(b2 / a0),
                  static_cast<float>(1.0), static_cast<float>(a1 / a0), static_cast<float>(a2 / a0));
  } else if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Biquad(input, output, static_cast<double>(b0 / a0), static_cast<double>(b1 / a0),
                  static_cast<double>(b2 / a0), static_cast<double>(1.0), static_cast<double>(a1 / a0),
                  static_cast<double>(a2 / a0));
  } else {
    return Biquad(input, output, static_cast<float16>(b0 / a0), static_cast<float16>(b1 / a0),
                  static_cast<float16>(b2 / a0), static_cast<float16>(1.0), static_cast<float16>(a1 / a0),
                  static_cast<float16>(a2 / a0));
  }
}
}  // namespace dataset
}  // namespace mindspore
