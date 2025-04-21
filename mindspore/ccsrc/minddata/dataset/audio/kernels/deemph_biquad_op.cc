/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
  RETURN_IF_NOT_OK(ValidateLowRank("DeemphBiquad", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorFloat("DeemphBiquad", input));
  const int32_t kSampleRate44100 = 44100;
  const int32_t kSampleRate48000 = 48000;
  int32_t central_freq = 0;
  double width_slope = 1;
  double gain = 0.0;
  if (sample_rate_ == kSampleRate44100) {
    central_freq = 5283;   // central_freq value from SoX
    width_slope = 0.4845;  // width_slope value from SoX
    gain = -9.477;         // gain value from SoX
  } else if (sample_rate_ == kSampleRate48000) {
    central_freq = 5356;  // central_freq value from SoX
    width_slope = 0.479;  // width_slope value from SoX
    gain = -9.62;         // gain value from SoX
  } else {
    RETURN_STATUS_UNEXPECTED(
      "The sample_rate parameter only supports 44100 or 48000, but got: " + std::to_string(sample_rate_) + ".");
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
