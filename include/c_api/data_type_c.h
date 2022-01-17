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
#ifndef MINDSPORE_INCLUDE_C_API_DATA_TYPE_C_H
#define MINDSPORE_INCLUDE_C_API_DATA_TYPE_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MSDataType {
  kMSDataTypeUnknown = 0,
  kMSDataTypeObjectTypeString = 12,
  kMSDataTypeObjectTypeList = 13,
  kMSDataTypeObjectTypeTuple = 14,
  kMSDataTypeObjectTypeTensor = 17,
  kMSDataTypeNumberTypeBegin = 29,
  kMSDataTypeNumberTypeBool = 30,
  kMSDataTypeNumberTypeInt8 = 32,
  kMSDataTypeNumberTypeInt16 = 33,
  kMSDataTypeNumberTypeInt32 = 34,
  kMSDataTypeNumberTypeInt64 = 35,
  kMSDataTypeNumberTypeUInt8 = 37,
  kMSDataTypeNumberTypeUInt16 = 38,
  kMSDataTypeNumberTypeUInt32 = 39,
  kMSDataTypeNumberTypeUInt64 = 40,
  kMSDataTypeNumberTypeFloat16 = 42,
  kMSDataTypeNumberTypeFloat32 = 43,
  kMSDataTypeNumberTypeFloat64 = 44,
  kMSDataTypeNumberTypeEnd = 46,
  // add new enum here
  kMSDataTypeInvalid = INT32_MAX,
} MSDataType;

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_INCLUDE_C_API_DATA_TYPE_C_H
