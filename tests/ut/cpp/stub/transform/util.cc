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

#include "include/transform/graph_ir/util.h"

#include <utility>
#include <map>

#include "securec/include/securec.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace transform {
const size_t kErrorSize = 0;
static std::map<MeDataType, size_t> datatype_size_map = {
  {MeDataType::kNumberTypeFloat16, sizeof(float) / 2}, {MeDataType::kNumberTypeFloat32, sizeof(float)},  // 1/2 of float
  {MeDataType::kNumberTypeFloat64, sizeof(double)},    {MeDataType::kNumberTypeInt8, sizeof(int8_t)},
  {MeDataType::kNumberTypeInt16, sizeof(int16_t)},     {MeDataType::kNumberTypeInt32, sizeof(int32_t)},
  {MeDataType::kNumberTypeInt64, sizeof(int64_t)},     {MeDataType::kNumberTypeUInt8, sizeof(uint8_t)},
  {MeDataType::kNumberTypeUInt16, sizeof(uint16_t)},   {MeDataType::kNumberTypeUInt32, sizeof(uint32_t)},
  {MeDataType::kNumberTypeUInt64, sizeof(uint64_t)},   {MeDataType::kNumberTypeBool, sizeof(bool)}};

size_t TransformUtil::GetDataTypeSize(const MeDataType &type) {
  if (datatype_size_map.find(type) != datatype_size_map.end()) {
    return datatype_size_map[type];
  } else {
    MS_LOG(ERROR) << "Illegal tensor data type!";
    return kErrorSize;
  }
}
}  // namespace transform
}  // namespace mindspore
