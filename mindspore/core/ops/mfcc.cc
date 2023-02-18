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

#include "ops/mfcc.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void Mfcc::Init(const float freq_upper_limit, const float freq_lower_limit, const int64_t filter_bank_channel_num,
                const int64_t dct_coeff_num) {
  this->set_freq_upper_limit(freq_upper_limit);
  this->set_freq_lower_limit(freq_lower_limit);
  this->set_filter_bank_channel_num(filter_bank_channel_num);
  this->set_dct_coeff_num(dct_coeff_num);
}

void Mfcc::set_freq_upper_limit(const float freq_upper_limit) {
  (void)this->AddAttr(kFreqUpperLimit, api::MakeValue(freq_upper_limit));
}

float Mfcc::get_freq_upper_limit() const {
  auto value_ptr = this->GetAttr(kFreqUpperLimit);
  return GetValue<float>(value_ptr);
}

void Mfcc::set_freq_lower_limit(const float freq_lower_limit) {
  (void)this->AddAttr(kFreqLowerLimit, api::MakeValue(freq_lower_limit));
}

float Mfcc::get_freq_lower_limit() const {
  auto value_ptr = this->GetAttr(kFreqLowerLimit);
  return GetValue<float>(value_ptr);
}

void Mfcc::set_filter_bank_channel_num(const int64_t filter_bank_channel_num) {
  (void)this->AddAttr(kFilterBankChannelNum, api::MakeValue(filter_bank_channel_num));
}

int64_t Mfcc::get_filter_bank_channel_num() const {
  auto value_ptr = this->GetAttr(kFilterBankChannelNum);
  return GetValue<int64_t>(value_ptr);
}

void Mfcc::set_dct_coeff_num(const int64_t dct_coeff_num) {
  (void)this->AddAttr(kDctCoeffNum, api::MakeValue(dct_coeff_num));
}

int64_t Mfcc::get_dct_coeff_num() const { return GetValue<int64_t>(GetAttr(kDctCoeffNum)); }

MIND_API_OPERATOR_IMPL(Mfcc, BaseOperator);
REGISTER_PRIMITIVE_C(kNameMfcc, Mfcc);
}  // namespace ops
}  // namespace mindspore
