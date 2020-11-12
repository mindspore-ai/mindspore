/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef LITE_MINDSPORE_LITE_C_OPS_MFCC_H_
#define LITE_MINDSPORE_LITE_C_OPS_MFCC_H_

#include <vector>
#include <set>
#include <cmath>
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class Mfcc : public PrimitiveC {
 public:
  Mfcc() = default;
  ~Mfcc() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(Mfcc, PrimitiveC);
  explicit Mfcc(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetFreqUpperLimit(float freq_upper_limit) {
    this->primitive_->value.AsMfcc()->freqUpperLimit = freq_upper_limit;
  }
  void SetFreqLowerLimit(float freq_lower_limit) {
    this->primitive_->value.AsMfcc()->freqLowerLimit = freq_lower_limit;
  }
  void SetFilterBankChannelNum(int filter_bank_channel_num) {
    this->primitive_->value.AsMfcc()->filterBankChannelNum = filter_bank_channel_num;
  }
  void SetDctCoeffNum(int dct_coeff_num) { this->primitive_->value.AsMfcc()->dctCoeffNum = dct_coeff_num; }
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  float GetFreqUpperLimit() const;
  float GetFreqLowerLimit() const;
  int GetFilterBankChannelNum() const;
  int GetDctCoeffNum() const;
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_MFCC_H_
