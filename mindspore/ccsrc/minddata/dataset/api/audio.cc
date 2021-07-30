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

#include "minddata/dataset/include/dataset/audio.h"

#include "minddata/dataset/audio/ir/kernels/band_biquad_ir.h"

namespace mindspore {
namespace dataset {

namespace audio {
// BandBiquad Transform Operation.
struct BandBiquad::Data {
  Data(int32_t sample_rate, float central_freq, float Q, bool noise)
      : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q), noise_(noise) {}
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
  bool noise_;
};

BandBiquad::BandBiquad(int32_t sample_rate, float central_freq, float Q, bool noise)
    : data_(std::make_shared<Data>(sample_rate, central_freq, Q, noise)) {}

std::shared_ptr<TensorOperation> BandBiquad::Parse() {
  return std::make_shared<BandBiquadOperation>(data_->sample_rate_, data_->central_freq_, data_->Q_, data_->noise_);
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
