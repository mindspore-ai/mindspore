/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ir/quantization_param.h"
#include "utils/ms_utils.h"

namespace mindspore {
bool QuantizationParam::operator==(const QuantizationParam &other) const {
  if (quant_algo_name() != other.quant_algo_name()) {
    return false;
  }
  return common::IsAttrsEqual(attrs_, other.attrs_);
}

bool QuantizationParam::operator==(const mindspore::Value &other) const {
  if (other.isa<QuantizationParam>()) {
    auto &other_prim = static_cast<const QuantizationParam &>(other);
    return *this == other_prim;
  } else {
    return false;
  }
}
}  // namespace mindspore
