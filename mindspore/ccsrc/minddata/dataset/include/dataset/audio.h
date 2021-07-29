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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/transforms.h"

namespace mindspore {
namespace dataset {

class TensorOperation;

// Transform operations for performing computer audio.
namespace audio {
/// \brief Design two-pole band filter.
class BandBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor (Default: 0.707).
  /// \param[in] noise Choose alternate mode for un-pitched audio or mode oriented to pitched audio(Default: False).
  explicit BandBiquad(int32_t sample_rate, float central_freq, float Q = 0.707, bool noise = false);

  /// \brief Destructor.
  ~BandBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

}  // namespace audio
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_
