/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_PROVIDERS_TRITON_BACKEND_SRC_MSLITE_UTILS_H_
#define MINDSPORE_LITE_TOOLS_PROVIDERS_TRITON_BACKEND_SRC_MSLITE_UTILS_H_

#include <map>
#include "include/api/data_type.h"
#include "triton/core/tritonserver.h"

namespace triton {
namespace backend {
namespace mslite {
static inline mindspore::DataType GetMSDataTypeFromTritonServerDataType(TRITONSERVER_DataType data_type) {
  static const std::map<TRITONSERVER_DataType, mindspore::DataType> ms_types = {
    {TRITONSERVER_TYPE_INVALID, mindspore::DataType::kInvalidType},
    {TRITONSERVER_TYPE_BOOL, mindspore::DataType::kNumberTypeBool},
    {TRITONSERVER_TYPE_UINT8, mindspore::DataType::kNumberTypeUInt8},
    {TRITONSERVER_TYPE_UINT16, mindspore::DataType::kNumberTypeUInt16},
    {TRITONSERVER_TYPE_UINT32, mindspore::DataType::kNumberTypeUInt32},
    {TRITONSERVER_TYPE_UINT64, mindspore::DataType::kNumberTypeUInt64},
    {TRITONSERVER_TYPE_INT8, mindspore::DataType::kNumberTypeInt8},
    {TRITONSERVER_TYPE_INT16, mindspore::DataType::kNumberTypeInt16},
    {TRITONSERVER_TYPE_INT32, mindspore::DataType::kNumberTypeInt32},
    {TRITONSERVER_TYPE_INT64, mindspore::DataType::kNumberTypeInt64},
    {TRITONSERVER_TYPE_FP16, mindspore::DataType::kNumberTypeFloat16},
    {TRITONSERVER_TYPE_FP32, mindspore::DataType::kNumberTypeFloat32},
    {TRITONSERVER_TYPE_FP64, mindspore::DataType::kNumberTypeFloat64},
    {TRITONSERVER_TYPE_BYTES, mindspore::DataType::kNumberTypeUInt8},
    {TRITONSERVER_TYPE_BF16, mindspore::DataType::kNumberTypeUInt16}};
  return ms_types.find(data_type) != ms_types.end() ? ms_types.at(data_type) : mindspore::DataType::kTypeUnknown;
}

static inline TRITONSERVER_DataType GetTritonServerDataTypeFromMSDataType(mindspore::DataType data_type) {
  static const std::map<mindspore::DataType, TRITONSERVER_DataType> triton_types = {
    {mindspore::DataType::kInvalidType, TRITONSERVER_TYPE_INVALID},
    {mindspore::DataType::kNumberTypeBool, TRITONSERVER_TYPE_BOOL},
    {mindspore::DataType::kNumberTypeUInt8, TRITONSERVER_TYPE_UINT8},
    {mindspore::DataType::kNumberTypeUInt16, TRITONSERVER_TYPE_UINT16},
    {mindspore::DataType::kNumberTypeUInt32, TRITONSERVER_TYPE_UINT32},
    {mindspore::DataType::kNumberTypeUInt64, TRITONSERVER_TYPE_UINT64},
    {mindspore::DataType::kNumberTypeInt8, TRITONSERVER_TYPE_INT8},
    {mindspore::DataType::kNumberTypeInt16, TRITONSERVER_TYPE_INT16},
    {mindspore::DataType::kNumberTypeInt32, TRITONSERVER_TYPE_INT32},
    {mindspore::DataType::kNumberTypeInt64, TRITONSERVER_TYPE_INT64},
    {mindspore::DataType::kNumberTypeFloat16, TRITONSERVER_TYPE_FP16},
    {mindspore::DataType::kNumberTypeFloat32, TRITONSERVER_TYPE_FP32},
    {mindspore::DataType::kNumberTypeFloat64, TRITONSERVER_TYPE_FP64},
  };
  return triton_types.find(data_type) != triton_types.end() ? triton_types.at(data_type) : TRITONSERVER_TYPE_INVALID;
}
}  // namespace mslite
}  // namespace backend
}  // namespace triton
#endif  // MINDSPORE_LITE_TOOLS_PROVIDERS_TRITON_BACKEND_SRC_MSLITE_UTILS_H_
