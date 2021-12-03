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

#include "minddata/dataset/audio/kernels/treble_biquad_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
TrebleBiquadOp::TrebleBiquadOp(int32_t sample_rate, float gain, float central_freq, float Q)
    : sample_rate_(sample_rate), gain_(gain), central_freq_(central_freq), Q_(Q) {}

Status TrebleBiquadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input tensor dimension, it should be greater than 0.
  RETURN_IF_NOT_OK(ValidateLowRank("TrebleBiquad", input, kMinAudioDim, "<..., time>"));
  // check input type, it should be DE_FLOAT32 or DE_FLOAT16 or DE_FLOAT64
  RETURN_IF_NOT_OK(ValidateTensorFloat("TrebleBiquad", input));
  // computer a0, a1, a2, b0, b1, b2
  float w0 = 2 * PI * central_freq_ / sample_rate_;
  float alpha = sin(w0) / 2 / Q_;
  // for peaking and shelving EQ filters only
  float attenuation = exp(gain_ / 40 * log(10));

  // temp1, temp2, temp3 are the intermediate variable used to solve for a and b.
  const float temp1 = 2 * sqrt(attenuation) * alpha;
  float temp2 = (attenuation - 1) * cos(w0);
  float temp3 = (attenuation + 1) * cos(w0);

  float b0 = attenuation * ((attenuation + 1) + temp2 + temp1);
  float b1 = -2 * attenuation * ((attenuation - 1) + temp3);
  float b2 = attenuation * ((attenuation + 1) + temp2 - temp1);
  float a0 = (attenuation + 1) - temp2 + temp1;
  float a1 = 2 * ((attenuation - 1) - temp3);
  float a2 = (attenuation + 1) - temp2 - temp1;
  // use Biquad function
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
