/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/audio/ir/kernels/pitch_shift_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/audio/kernels/pitch_shift_op.h"
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace audio {
// PitchShift
PitchShiftOperation::PitchShiftOperation(int32_t sample_rate, int32_t n_steps, int32_t bins_per_octave, int32_t n_fft,
                                         int32_t win_length, int32_t hop_length, WindowType window)
    : sample_rate_(sample_rate),
      n_steps_(n_steps),
      bins_per_octave_(bins_per_octave),
      n_fft_(n_fft),
      win_length_(win_length),
      hop_length_(hop_length),
      window_(window) {}

PitchShiftOperation::~PitchShiftOperation() = default;

Status PitchShiftOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("PitchShift", "n_fft", n_fft_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("PitchShift", "win_length", win_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("PitchShift", "hop_length", hop_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("PitchShift", "sample_rate", sample_rate_));
  RETURN_IF_NOT_OK(ValidateScalarNotZero("PitchShift", "bins_per_octave", bins_per_octave_));
  CHECK_FAIL_RETURN_UNEXPECTED(win_length_ <= n_fft_,
                               "PitchShift: win_length must be less than or equal to n_fft, but got win_length: " +
                                 std::to_string(win_length_) + ", n_fft: " + std::to_string(n_fft_));
  return Status::OK();
}

std::shared_ptr<TensorOp> PitchShiftOperation::Build() {
  win_length_ = win_length_ == 0 ? n_fft_ : win_length_;
  constexpr int kRateSize = 4;
  hop_length_ = hop_length_ == 0 ? (win_length_ / kRateSize) : hop_length_;
  std::shared_ptr<PitchShiftOp> tensor_op =
    std::make_shared<PitchShiftOp>(sample_rate_, n_steps_, bins_per_octave_, n_fft_, win_length_, hop_length_, window_);
  return tensor_op;
}

std::string PitchShiftOperation::Name() const { return kPitchShiftOperation; }

Status PitchShiftOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["n_steps"] = n_steps_;
  args["bins_per_octave"] = bins_per_octave_;
  args["n_fft"] = n_fft_;
  args["win_length"] = win_length_;
  args["hop_length"] = hop_length_;
  args["window"] = window_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
