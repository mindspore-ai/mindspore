/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "ir/dtype/type_id.h"
#include <map>

namespace mindspore {
namespace abstract {
const std::map<TypeId, size_t> type_map = {
  {kNumberTypeBool, 1},        {kNumberTypeInt, 4},      {kNumberTypeInt8, 1},          {kNumberTypeInt16, 2},
  {kNumberTypeInt32, 4},       {kNumberTypeInt64, 8},    {kNumberTypeUInt, 4},          {kNumberTypeUInt8, 1},
  {kNumberTypeUInt16, 2},      {kNumberTypeUInt32, 4},   {kNumberTypeUInt64, 8},        {kNumberTypeFloat, 4},
  {kNumberTypeFloat16, 2},     {kNumberTypeFloat32, 4},  {kNumberTypeFloat64, 8},       {kNumberTypeComplex64, 8},
  {kNumberTypeComplex128, 16}, {kNumberTypeBFloat16, 2}, {kNumberTypeInt4, 1},          {kNumberTypeFloat8E4M3FN, 1},
  {kNumberTypeFloat8E5M2, 1},  {kNumberTypeHiFloat8, 1}, {kNumberTypeFloat8E8M0FNU, 1}, {kObjectTypeString, 1}};
size_t TypeIdSize(TypeId data_type) {
  const size_t unsupported_type_error = 0;
  auto iter = type_map.find(data_type);
  if (iter != type_map.end()) {
    return iter->second;
  }
  return unsupported_type_error;
}
}  // namespace abstract
}  // namespace mindspore
