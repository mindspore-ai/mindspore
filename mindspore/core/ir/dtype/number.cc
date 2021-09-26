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

namespace mindspore {
bool Number::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  auto other_number = static_cast<const Number &>(other);
  return ((number_type_ == other_number.number_type_) && (nbits_ == other_number.nbits_));
}

Int::Int(const int nbits) : Number(IntBitsToTypeId(nbits), nbits, false) {
  if (nbits != static_cast<int>(BitsNum::eBits8) && nbits != static_cast<int>(BitsNum::eBits16) &&
      nbits != static_cast<int>(BitsNum::eBits32) && nbits != static_cast<int>(BitsNum::eBits64)) {
    MS_LOG(EXCEPTION) << "Wrong number of bits.";
  }
}

UInt::UInt(const int nbits) : Number(UIntBitsToTypeId(nbits), nbits, false) {
  if (nbits != static_cast<int>(BitsNum::eBits8) && nbits != static_cast<int>(BitsNum::eBits16) &&
      nbits != static_cast<int>(BitsNum::eBits32) && nbits != static_cast<int>(BitsNum::eBits64)) {
    MS_LOG(EXCEPTION) << "Wrong number of bits.";
  }
}

Float::Float(const int nbits) : Number(FloatBitsToTypeId(nbits), nbits, false) {
  if (nbits != static_cast<int>(BitsNum::eBits16) && nbits != static_cast<int>(BitsNum::eBits32) &&
      nbits != static_cast<int>(BitsNum::eBits64)) {
    MS_LOG(EXCEPTION) << "Wrong number of bits.";
  }
}

Complex::Complex(const int nbits) : Number(ComplexBitsToTypeId(nbits), nbits, false) {
  if (nbits != static_cast<int>(BitsNum::eBits64) && nbits != static_cast<int>(BitsNum::eBits128)) {
    MS_LOG(EXCEPTION) << "Wrong number of bits.";
  }
}
}  // namespace mindspore
