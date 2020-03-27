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
#include "kernel/aicpu/aicpu_util.h"
#include <vector>
#include <string>
#include "proto/types.pb.h"
#include "runtime/mem.h"
#include "runtime/rt.h"
#include "utils/convert_utils.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
static std::map<int32_t, int32_t> MS_PROTO_DATA_TYPE_MAP = {
  {mindspore::TypeId::kTypeUnknown, mindspore::DataType::MS_UNKNOWN},
  {mindspore::TypeId::kNumberTypeBool, mindspore::DataType::MS_BOOL},
  {mindspore::TypeId::kNumberTypeInt8, mindspore::DataType::MS_INT8},
  {mindspore::TypeId::kNumberTypeInt16, mindspore::DataType::MS_INT16},
  {mindspore::TypeId::kNumberTypeInt32, mindspore::DataType::MS_INT32},
  {mindspore::TypeId::kNumberTypeInt64, mindspore::DataType::MS_INT64},
  {mindspore::TypeId::kNumberTypeUInt, mindspore::DataType::MS_UINT32},
  {mindspore::TypeId::kNumberTypeUInt8, mindspore::DataType::MS_UINT8},
  {mindspore::TypeId::kNumberTypeUInt16, mindspore::DataType::MS_UINT16},
  {mindspore::TypeId::kNumberTypeUInt64, mindspore::DataType::MS_UINT64},
  {mindspore::TypeId::kNumberTypeFloat16, mindspore::DataType::MS_FLOAT16},
  {mindspore::TypeId::kNumberTypeFloat32, mindspore::DataType::MS_FLOAT32},
  {mindspore::TypeId::kNumberTypeFloat64, mindspore::DataType::MS_FLOAT64},
};

int AicpuOpUtil::MsTypeToProtoType(TypeId ms_type) {
  auto iter = MS_PROTO_DATA_TYPE_MAP.find(ms_type);
  if (iter != MS_PROTO_DATA_TYPE_MAP.end()) {
    return MS_PROTO_DATA_TYPE_MAP[ms_type];
  } else {
    MS_LOG(ERROR) << "UnSupported ms_type value" << static_cast<int>(ms_type);
    return -1;
  }
}
}  // namespace kernel
}  // namespace mindspore
