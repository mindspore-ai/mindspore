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
#include "minddata/dataset/audio/kernels/riaa_biquad_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
RiaaBiquadOp::RiaaBiquadOp(int32_t sample_rate) : sample_rate_(sample_rate) {}

Status RiaaBiquadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input tensor dimension, it should be greater than 0.
  RETURN_IF_NOT_OK(ValidateLowRank("RiaaBiquad", input, kMinAudioDim, "<..., time>"));
  // check input type, it should be DE_FLOAT32 or DE_FLOAT16 or DE_FLOAT64.
  RETURN_IF_NOT_OK(ValidateTensorFloat("RiaaBiquad", input));
  // indicate array zeros and poles.
  const std::map<int32_t, std::vector<float>> kZeros = {
    {44100, {-0.2014898, 0.9233820}},
    {48000, {-0.1766069, 0.9321590}},
    {88200, {-0.1168735, 0.9648312}},
    {96000, {-0.1141486, 0.9676817}},
  };
  const std::map<int32_t, std::vector<float>> kPoles = {
    {44100, {0.7083149, 0.9924091}},
    {48000, {0.7396325, 0.9931330}},
    {88200, {0.8590646, 0.9964002}},
    {96000, {0.8699137, 0.9966946}},
  };
  const std::vector<float> &zeros = kZeros.at(sample_rate_);
  const std::vector<float> &poles = kPoles.at(sample_rate_);
  // computer a0, a1, a2, b0, b1, b2.
  // polynomial coefficients with roots zeros[0] and zeros[1].
  float b0 = 1.0;
  float b1 = -(zeros[0] + zeros[1]);
  float b2 = zeros[0] * zeros[1];
  // polynomial coefficients with roots poles[0] and poles[1].
  float a0 = 1.0;
  float a1 = -(poles[0] + poles[1]);
  float a2 = poles[0] * poles[1];
  // normalize to 0dB at 1kHz.
  float w0 = 2 * PI * 1000 / sample_rate_;
  // re refers to the real part of the complex number.
  float b_re = b0 + b1 * cos(-w0) + b2 * cos(-2 * w0);
  float a_re = a0 + a1 * cos(-w0) + a2 * cos(-2 * w0);
  // im refers to the imaginary part of the complex number.
  float b_im = b1 * sin(-w0) + b2 * sin(-2 * w0);
  float a_im = a1 * sin(-w0) + a2 * sin(-2 * w0);
  // temp is the intermediate variable used to solve for b0, b1, b2.
  const float temp = 1 / sqrt((b_re * b_re + b_im * b_im) / (a_re * a_re + a_im * a_im));
  b0 *= temp;
  b1 *= temp;
  b2 *= temp;
  // use Biquad function.
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
