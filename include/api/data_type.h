/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_DATA_TYPE_H_
#define MINDSPORE_INCLUDE_API_DATA_TYPE_H_

#include <cstdint>

namespace mindspore {
enum class DataType : int {
  kTypeUnknown = 0,
  kObjectTypeString = 12,
  kObjectTypeList = 13,
  kObjectTypeTuple = 14,
  kObjectTypeTensorType = 17,
  kNumberTypeBegin = 29,
  kNumberTypeBool = 30,
  kNumberTypeInt8 = 32,
  kNumberTypeInt16 = 33,
  kNumberTypeInt32 = 34,
  kNumberTypeInt64 = 35,
  kNumberTypeUInt8 = 37,
  kNumberTypeUInt16 = 38,
  kNumberTypeUInt32 = 39,
  kNumberTypeUInt64 = 40,
  kNumberTypeFloat16 = 42,
  kNumberTypeFloat32 = 43,
  kNumberTypeFloat64 = 44,
  kNumberTypeBFloat16 = 46,
  kNumberTypeEnd = 53,
  // add new enum here
  kInvalidType = INT32_MAX,
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_DATA_TYPE_H_
