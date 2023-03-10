/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_GE_DUMP_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_GE_DUMP_H_

#include <map>
#include <string>
#include "proto/ge_dtype.pb.h"
#include "ir/dtype/type_id.h"
#include "include/common/utils/utils.h"
#include "external/graph/types.h"
#include "hccl/hccl_types.h"

namespace mindspore {
namespace device {
namespace ascend {
class GeTypesConvert {
 public:
  GeTypesConvert() = default;
  ~GeTypesConvert() = default;
  static ::ge::proto::DataType GetGeDataType(TypeId type_id);
  static ::ge::proto::DataType TransHcclDataTypeToGeDataType(HcclDataType dtype);
  static ::ge::Format GetGeFormat(const std::string &format, size_t shape_size);
  static std::string GetGeTilingFormat(::ge::Format ge_format);
  static ::ge::DataType TransTypeIdToGeDataType(TypeId type_id);
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_GE_DUMP_H_
