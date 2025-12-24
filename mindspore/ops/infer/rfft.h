/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_RFFT_H_
#define MINDSPORE_CORE_OPS_RFFT_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRfft = "Rfft";
/// \brief Rfft defined the operator prototype of computing discrete fourier transform of a real-valued signal.
class OPS_API Rfft : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Rfft);
  /// \brief Constructor.
  Rfft() : BaseOperator(kNameRfft) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] fft_length Define the FFT length to compute discrete fourier transform.
  void Init(const int64_t fft_length);

  /// \brief Method to set fft_length attribute.
  ///
  /// \param[in] fft_length Define the FFT length to compute discrete fourier transform.
  void set_fft_length(const int64_t fft_length);

  /// \brief Method to get fft_length attribute.
  ///
  /// \return the FFT length.
  int64_t get_fft_length() const;
};
OPS_API abstract::AbstractBasePtr RfftInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RFFT_H_
