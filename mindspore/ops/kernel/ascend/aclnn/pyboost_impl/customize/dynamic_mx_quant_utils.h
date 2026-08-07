/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_DYNAMIC_MX_QUANT_UTILS_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_DYNAMIC_MX_QUANT_UTILS_H_

#include <string>
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore::kernel::pyboost::dynamic_mx_quant_utils {
inline aclDataType ConvertDstType(int64_t dst_type) {
  switch (dst_type) {
    case 23:
    case 291:
      return ACL_FLOAT8_E5M2;
    case 24:
    case 292:
      return ACL_FLOAT8_E4M3FN;
    case 296:
      return ACL_FLOAT4_E2M1;
    case 297:
      return ACL_FLOAT4_E1M2;
    default:
      MS_LOG(EXCEPTION) << "Unsupported dynamic MX quant destination type identifier: " << dst_type << ".";
  }
  return ACL_DT_UNDEFINED;
}

inline bool IsAclnnApiAvailable(const std::string &api_name) {
  const auto workspace_api_name = api_name + "GetWorkspaceSize";
  return device::ascend::GetOpApiFunc(api_name.c_str()) != nullptr &&
         device::ascend::GetOpApiFunc(workspace_api_name.c_str()) != nullptr;
}
}  // namespace mindspore::kernel::pyboost::dynamic_mx_quant_utils

#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_DYNAMIC_MX_QUANT_UTILS_H_
