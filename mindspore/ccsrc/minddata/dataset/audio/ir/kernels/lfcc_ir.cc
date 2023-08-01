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
#include "minddata/dataset/audio/ir/kernels/lfcc_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/audio/kernels/lfcc_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
LFCCOperation::LFCCOperation(int32_t sample_rate, int32_t n_filter, int32_t n_lfcc, float f_min, float f_max,
                             int32_t dct_type, NormMode norm, bool log_lf, int32_t n_fft, int32_t win_length,
                             int32_t hop_length, int32_t pad, WindowType window, float power, bool normalized,
                             bool center, BorderType pad_mode, bool onesided)
    : sample_rate_(sample_rate),
      n_filter_(n_filter),
      n_lfcc_(n_lfcc),
      f_min_(f_min),
      f_max_(f_max),
      dct_type_(dct_type),
      norm_(norm),
      log_lf_(log_lf),
      n_fft_(n_fft),
      win_length_(win_length),
      hop_length_(hop_length),
      pad_(pad),
      window_(window),
      power_(power),
      normalized_(normalized),
      center_(center),
      pad_mode_(pad_mode),
      onesided_(onesided) {}

LFCCOperation::~LFCCOperation() = default;

std::string LFCCOperation::Name() const { return kLFCCOperation; }

Status LFCCOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("LFCC", "sample_rate", sample_rate_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("LFCC", "n_filter", n_filter_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("LFCC", "n_lfcc", n_lfcc_));
  CHECK_FAIL_RETURN_UNEXPECTED(dct_type_ == TWO,
                               "LFCC: dct_type must be equal to 2, but got: " + std::to_string(dct_type_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("LFCC", "f_max", f_max_));
  CHECK_FAIL_RETURN_UNEXPECTED(n_lfcc_ <= n_fft_,
                               "LFCC: n_fft should be greater than or equal to n_lfcc, but got n_lfcc: " +
                                 std::to_string(n_lfcc_) + " and n_fft: " + std::to_string(n_fft_));

  // Spectrogram params
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("LFCC", "n_fft", n_fft_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("LFCC", "win_length", win_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("LFCC", "hop_length", hop_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("LFCC", "pad", pad_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("LFCC", "power", power_));
  CHECK_FAIL_RETURN_UNEXPECTED(pad_mode_ != BorderType::kEdge, "LFCC: invalid BorderType, kEdge is not supported.");
  if (f_max_ != 0) {
    RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("LFCC", "f_max", f_max_));
    CHECK_FAIL_RETURN_UNEXPECTED(
      f_min_ <= f_max_, "LFCC: f_max must be greater than or equal to f_min, but got f_max: " + std::to_string(f_max_) +
                          " and f_min: " + std::to_string(f_min_));
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(f_min_ < (sample_rate_ / TWO),
                                 "LFCC: f_min must be less than half of sample_rate when f_max is 0, but got f_min: " +
                                   std::to_string(f_min_) + " and sample_rate: " + std::to_string(sample_rate_));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(win_length_ <= n_fft_,
                               "LFCC: win_length must be less than or equal to n_fft, but got win_length: " +
                                 std::to_string(win_length_) + ", n_fft: " + std::to_string(n_fft_));
  return Status::OK();
}

std::shared_ptr<TensorOp> LFCCOperation::Build() {
  win_length_ = win_length_ == 0 ? n_fft_ : win_length_;
  hop_length_ = hop_length_ == 0 ? (win_length_ / TWO) : hop_length_;
  f_max_ = f_max_ == 0 ? floor(sample_rate_ / TWO) : f_max_;
  std::shared_ptr<LFCCOp> tensor_op =
    std::make_shared<LFCCOp>(sample_rate_, n_filter_, n_lfcc_, dct_type_, log_lf_, n_fft_, win_length_, hop_length_,
                             f_min_, f_max_, pad_, window_, power_, normalized_, center_, pad_mode_, onesided_, norm_);
  return tensor_op;
}

Status LFCCOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["n_filter"] = n_filter_;
  args["n_lfcc"] = n_lfcc_;
  args["f_min"] = f_min_;
  args["f_max"] = f_max_;
  args["dct_type"] = dct_type_;
  args["norm"] = norm_;
  args["log_lf"] = log_lf_;
  args["n_fft"] = n_fft_;
  args["win_length"] = win_length_;
  args["hop_length"] = hop_length_;
  args["pad"] = pad_;
  args["window"] = window_;
  args["power"] = power_;
  args["normalized"] = normalized_;
  args["center"] = center_;
  args["pad_mode"] = pad_mode_;
  args["onesided"] = onesided_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
