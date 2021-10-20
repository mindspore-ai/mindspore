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
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMfcc = "Mfcc";
/// \brief Mfcc defined the operator prototype of extracting Mel-Frequency Cepstral Coefficients.
class MS_CORE_API Mfcc : public PrimitiveC {
 public:
  /// \brief Constructor.
  Mfcc() : PrimitiveC(kNameMfcc) {}

  /// \brief Destructor.
  ~Mfcc() = default;

  MS_DECLARE_PARENT(Mfcc, PrimitiveC);

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
AbstractBasePtr MfccInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args);
using PrimMfccPtr = std::shared_ptr<Mfcc>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MFCC_H_
