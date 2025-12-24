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
#ifndef MINDSPORE_CORE_OPS_MFCC_H_
#define MINDSPORE_CORE_OPS_MFCC_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMfcc = "Mfcc";
/// \brief Mfcc defined the operator prototype of extracting Mel-Frequency Cepstral Coefficients.
class OPS_API Mfcc : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Mfcc);
  /// \brief Constructor.
  Mfcc() : BaseOperator(kNameMfcc) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] freq_upper_limit Define the highest frequency to use when computing the cepstrum.
  /// \param[in] freq_lower_limit Define the lowest frequency to use when computing the cepstrum.
  /// \param[in] filter_bank_channel_num Define the count of mel bank to use internally.
  /// \param[in] dct_coeff_num Define the output channels to generate per time slice.
  void Init(const float freq_upper_limit, const float freq_lower_limit, const int64_t filter_bank_channel_num,
            const int64_t dct_coeff_num);

  /// \brief Method to set freq_upper_limit attribute.
  ///
  /// \param[in] freq_upper_limit Define the highest frequency to use when computing the cepstrum.
  void set_freq_upper_limit(const float freq_upper_limit);

  /// \brief Method to set freq_lower_limit attribute.
  ///
  /// \param[in] freq_lower_limit Define the lowest frequency to use when computing the cepstrum.
  void set_freq_lower_limit(const float freq_lower_limit);

  /// \brief Method to set filter_bank_channel_num attribute.
  ///
  /// \param[in] filter_bank_channel_num Define the count of mel bank to use internally.
  void set_filter_bank_channel_num(const int64_t filter_bank_channel_num);

  /// \brief Method to set dct_coeff_num attribute.
  ///
  /// \param[in] dct_coeff_num Define the output channels to generate per time slice.
  void set_dct_coeff_num(const int64_t dct_coeff_num);

  /// \brief Method to get freq_upper_limit attribute.
  ///
  /// \return the highest frequency.
  float get_freq_upper_limit() const;

  /// \brief Method to get freq_lower_limit attribute.
  ///
  /// \return the lowest frequency.
  float get_freq_lower_limit() const;

  /// \brief Method to get filter_bank_channel_num attribute.
  ///
  /// \return  the count of mel bank to use internally.
  int64_t get_filter_bank_channel_num() const;

  /// \brief Method to get dct_coeff_num attribute.
  ///
  /// \return the output channels to generate per time slice.
  int64_t get_dct_coeff_num() const;
};
OPS_API abstract::AbstractBasePtr MfccInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MFCC_H_
