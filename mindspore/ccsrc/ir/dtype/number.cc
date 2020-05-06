/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "ir/dtype/number.h"
#include <string>
#include <cstdlib>
#include <algorithm>
#include "utils/log_adapter.h"
#include "pipeline/static_analysis/abstract_value.h"
#include "pybind_api/api_register.h"
#include "pybind_api/export_flags.h"

namespace mindspore {
bool Number::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  auto other_number = static_cast<const Number &>(other);
  return ((number_type_ == other_number.number_type_) && (nbits_ == other_number.nbits_));
}

Int::Int(const int nbits) : Number(IntBitsToTypeId(nbits), nbits, false) {
  if (nbits != 8 && nbits != 16 && nbits != 32 && nbits != 64) {
    MS_LOG(EXCEPTION) << "Wrong number of bits.";
  }
}

UInt::UInt(const int nbits) : Number(UIntBitsToTypeId(nbits), nbits, false) {
  if (nbits != 8 && nbits != 16 && nbits != 32 && nbits != 64) {
    MS_LOG(EXCEPTION) << "Wrong number of bits.";
  }
}

Float::Float(const int nbits) : Number(FloatBitsToTypeId(nbits), nbits, false) {
  if (nbits != 16 && nbits != 32 && nbits != 64) {
    MS_LOG(EXCEPTION) << "Wrong number of bits.";
  }
}

const TypePtr kBool = std::make_shared<Bool>();
const TypePtr kInt8 = std::make_shared<Int>(8);
const TypePtr kInt16 = std::make_shared<Int>(16);
const TypePtr kInt32 = std::make_shared<Int>(32);
const TypePtr kInt64 = std::make_shared<Int>(64);
const TypePtr kUInt8 = std::make_shared<UInt>(8);
const TypePtr kUInt16 = std::make_shared<UInt>(16);
const TypePtr kUInt32 = std::make_shared<UInt>(32);
const TypePtr kUInt64 = std::make_shared<UInt>(64);
const TypePtr kFloat16 = std::make_shared<Float>(16);
const TypePtr kFloat32 = std::make_shared<Float>(32);
const TypePtr kFloat64 = std::make_shared<Float>(64);
const TypePtr kInt = std::make_shared<Int>();
const TypePtr kUInt = std::make_shared<UInt>();
const TypePtr kFloat = std::make_shared<Float>();
const TypePtr kNumber = std::make_shared<Number>();
}  // namespace mindspore
