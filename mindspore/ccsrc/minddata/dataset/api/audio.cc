/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/audio/transform/validators.h"
#include "minddata/dataset/audio/kernels/audio_utils.h"

namespace mindspore {
namespace dataset {
namespace audio {
Status CreateDct(mindspore::MSTensor *output, int32_t n_mfcc, int32_t n_mels, NormMode norm) {
  RETURN_UNEXPECTED_IF_NULL(output);
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("CreateDct", "n_mfcc", n_mfcc));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("CreateDct", "n_mels", n_mels));

  std::shared_ptr<dataset::Tensor> dct;
  RETURN_IF_NOT_OK(Dct(&dct, n_mfcc, n_mels, norm));
  CHECK_FAIL_RETURN_UNEXPECTED(dct->HasData(), "CreateDct: get an empty tensor with shape " + dct->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(dct));
  return Status::OK();
}

// MelscaleFbanks Function.
Status MelscaleFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_mels, int32_t sample_rate,
                      NormType norm, MelType mel_type) {
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(n_freqs > 0,
                               "MelscaleFbanks: n_freqs must be greater than 0, got: " + std::to_string(n_freqs));

  CHECK_FAIL_RETURN_UNEXPECTED(f_min >= 0, "MelscaleFbanks: f_min must be non negative, got: " + std::to_string(f_min));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > 0,
                               "MelscaleFbanks: f_max must be greater than 0, got: " + std::to_string(f_max));
  CHECK_FAIL_RETURN_UNEXPECTED(n_mels > 0,
                               "MelscaleFbanks: n_mels must be greater than 0, got: " + std::to_string(n_mels));
  CHECK_FAIL_RETURN_UNEXPECTED(
    sample_rate > 0, "MelscaleFbanks: sample_rate must be greater than 0, got: " + std::to_string(sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > f_min, "MelscaleFbanks: f_max must be greater than f_min, got: f_min = " +
                                                std::to_string(f_min) + ", while f_max = " + std::to_string(f_max));
  std::shared_ptr<dataset::Tensor> fb;
  RETURN_IF_NOT_OK(CreateFbanks<float>(&fb, n_freqs, f_min, f_max, n_mels, sample_rate, norm, mel_type));
  CHECK_FAIL_RETURN_UNEXPECTED(fb->HasData(),
                               "MelscaleFbanks: get an empty tensor with shape " + fb->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(fb));
  return Status::OK();
}

// LinearFbanks Function.
Status LinearFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_filter,
                    int32_t sample_rate) {
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(n_freqs > 0,
                               "LinearFbanks: n_freqs must be greater than 0, got: " + std::to_string(n_freqs));

  CHECK_FAIL_RETURN_UNEXPECTED(f_min >= 0, "LinearFbanks: f_min must be non negative, got: " + std::to_string(f_min));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > 0, "LinearFbanks: f_max must be greater than 0, got: " + std::to_string(f_max));
  CHECK_FAIL_RETURN_UNEXPECTED(n_filter > 0,
                               "LinearFbanks: n_filter must be greater than 0, got: " + std::to_string(n_filter));
  CHECK_FAIL_RETURN_UNEXPECTED(sample_rate > 0,
                               "LinearFbanks: sample_rate must be greater than 0, got: " + std::to_string(sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > f_min, "LinearFbanks: f_max must be greater than f_min, got: f_min = " +
                                                std::to_string(f_min) + ", while f_max = " + std::to_string(f_max));
  std::shared_ptr<dataset::Tensor> fb;
  RETURN_IF_NOT_OK(CreateLinearFbanks(&fb, n_freqs, f_min, f_max, n_filter, sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(fb->HasData(), "LinearFbanks: get an empty tensor with shape " + fb->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(fb));
  return Status::OK();
}

}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
