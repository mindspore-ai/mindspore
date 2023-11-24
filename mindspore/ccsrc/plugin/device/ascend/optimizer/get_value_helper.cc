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

#include "plugin/device/ascend/optimizer/get_value_helper.h"
#include "mindapi/base/format.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
std::string GetNodeFormatValue(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto format_value = node->abstract()->GetValue();
  MS_EXCEPTION_IF_NULL(format_value);
  auto format_enum = static_cast<mindspore::Format>(ops::GetValueWithCheck<int64_t>(format_value));
  auto format = FormatEnumToString(format_enum);
  return format;
}

template <typename T>
T GetNodeScalarValue(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_ptr = node->abstract()->GetValue();
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto value = ops::GetValueWithCheck<T>(value_ptr);
  return value;
}

template bool GetNodeScalarValue(const AnfNodePtr &node);

template float GetNodeScalarValue(const AnfNodePtr &node);

template double GetNodeScalarValue(const AnfNodePtr &node);

template int32_t GetNodeScalarValue(const AnfNodePtr &node);

template int64_t GetNodeScalarValue(const AnfNodePtr &node);
}  // namespace opt
}  // namespace mindspore
