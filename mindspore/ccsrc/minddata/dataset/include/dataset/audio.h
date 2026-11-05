/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "utils/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/ms_tensor.h"
#include "include/dataset/transforms.h"
#include "minddata/dataset/util/visible.h"

namespace mindspore::dataset {
class TensorOperation;

// Transform operations for performing computer audio.
namespace audio {

/// \param[in] n_mfcc Number of mfc coefficients to retain, the value must be greater than 0.
/// \param[in] n_mels Number of mel filterbanks, the value must be greater than 0.
/// \param[in] norm Norm to use, can be NormMode::kNone or NormMode::kOrtho.
/// \return Status error code, returns OK if no error encountered.
Status CreateDct(mindspore::MSTensor *output, int32_t n_mfcc, int32_t n_mels, NormMode norm = NormMode::kNone);

/// \brief Creates a linear triangular filterbank.
/// \param output Tensor of a linear triangular filterbank.
/// \param n_freqs: Number of frequency.
/// \param f_min: Minimum of frequency in Hz.
/// \param f_max: Maximum of frequency in Hz.
/// \param n_filter: Number of (linear) triangular filter.
/// \param sample_rate: Sample rate.
/// \return Status code.
Status DATASET_API LinearFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_filter,
                                int32_t sample_rate);

/// \brief Create a frequency transformation matrix with shape (n_freqs, n_mels).
/// \param[in] output Tensor of the frequency transformation matrix.
/// \param[in] n_freqs Number of frequencies to highlight/apply.
/// \param[in] f_min Minimum frequency (Hz).
/// \param[in] f_max Maximum frequency (Hz).
/// \param[in] n_mels Number of mel filterbanks.
/// \param[in] sample_rate Sample rate of the audio waveform.
/// \param[in] norm Norm to use, can be NormType::kNone or NormType::kSlaney. Default: NormType::kNone.
/// \param[in] mel_type Scale to use, can be MelType::kHtk or MelType::kSlaney. Default: MelType::kHtz.
/// \return Status code.
Status DATASET_API MelscaleFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_mels,
                                  int32_t sample_rate, NormType norm = NormType::kNone,
                                  MelType mel_type = MelType::kHtk);

// \brief Shift the pitch of a waveform by 'n_steps' steps.

}  // namespace audio
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_
