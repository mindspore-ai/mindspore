/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "mindspore/core/ir/dtype/type_utils.h"

#include <algorithm>
#include <cstdlib>
#include <climits>

#include "ir/dtype/type.h"
#include "utils/log_adapter.h"

namespace mindspore {
TypeId IntBitsToTypeId(const int nbits) {
  switch (nbits) {
    case static_cast<int>(BitsNum::eBits4):
      return kNumberTypeInt4;
    case static_cast<int>(BitsNum::eBits8):
      return kNumberTypeInt8;
    case static_cast<int>(BitsNum::eBits16):
      return kNumberTypeInt16;
    case static_cast<int>(BitsNum::eBits32):
      return kNumberTypeInt32;
    case static_cast<int>(BitsNum::eBits64):
      return kNumberTypeInt64;
    default:
      MS_LOG(EXCEPTION) << "For Int type only support number of 8bits, 16bits, 32bits and 64bits, but got " << nbits
                        << "bits";
  }
}

TypeId UIntBitsToTypeId(const int nbits) {
  switch (nbits) {
    case static_cast<int>(BitsNum::eBits8):
      return kNumberTypeUInt8;
    case static_cast<int>(BitsNum::eBits16):
      return kNumberTypeUInt16;
    case static_cast<int>(BitsNum::eBits32):
      return kNumberTypeUInt32;
    case static_cast<int>(BitsNum::eBits64):
      return kNumberTypeUInt64;
    default:
      MS_LOG(EXCEPTION) << "For UInt type only support number of 8bits, 16bits, 32bits and 64bits, but got " << nbits
                        << "bits";
  }
}

TypeId FloatBitsToTypeId(const int nbits) {
  switch (nbits) {
    case static_cast<int>(BitsNum::eBits16):
      return kNumberTypeFloat16;
    case static_cast<int>(BitsNum::eBits32):
      return kNumberTypeFloat32;
    case static_cast<int>(BitsNum::eBits64):
      return kNumberTypeFloat64;
    default:
      MS_LOG(EXCEPTION) << "For Float type only support number of 16bits, 32bits and 64bits, but got " << nbits
                        << "bits";
  }
}

TypeId BFloatBitsToTypeId(const int nbits) {
  switch (nbits) {
    case static_cast<int>(BitsNum::eBits16):
      return kNumberTypeBFloat16;
    default:
      MS_LOG(EXCEPTION) << "For BFloat type only support number of 16bits, but got " << nbits << "bits";
  }
}

TypeId ComplexBitsToTypeId(const int nbits) {
  switch (nbits) {
    case static_cast<int>(BitsNum::eBits64):
      return kNumberTypeComplex64;
    case static_cast<int>(BitsNum::eBits128):
      return kNumberTypeComplex128;
    default:
      MS_LOG(EXCEPTION) << "For Complex type only support number of 64bits and 128bits, but got " << nbits << "bits";
  }
}

bool IsSameObjectType(const Type &lhs, const Type &rhs) {
  if ((lhs.meta_type() != kMetaTypeObject) || (rhs.meta_type() != kMetaTypeObject)) {
    return false;
  }
  return lhs.object_type() == rhs.object_type();
}
}  // namespace mindspore
