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

#include "kernel/ascend/acl_ir/acl_adapter_info.h"
#include "include/utils/utils.h"

namespace mindspore::device::ascend {
std::string CheckNdSupported(TypeId data_type, const std::vector<ShapeVector> &shapes) {
  constexpr size_t special_mm_size = 2;
  if (shapes.size() != special_mm_size || shapes[0].size() != special_mm_size || shapes[1].size() != special_mm_size) {
    return kOpFormat_DEFAULT;
  }
  if (data_type != kNumberTypeFloat16) {
    return kOpFormat_DEFAULT;
  }
  auto is_align = [&shapes]() {
    return (!(static_cast<uint64_t>(shapes[0][0]) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(shapes[0][1]) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(shapes[1][0]) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(shapes[1][1]) & 0x0000000F));
  };
  if (is_align()) {
    return kOpFormat_DEFAULT;
  }
  return kOpFormat_FRAC_NZ;
}

REGISTER_ACL_OP(MatMulV2).OutputSelector(&CheckNdSupported);
REGISTER_ACL_OP(MatMul).OutputSelector(&CheckNdSupported);

std::string CheckBMMNdSupported(TypeId data_type, const std::vector<ShapeVector> &shapes) {
  constexpr size_t special_bmm_size = 2;
  if (shapes.size() != special_bmm_size) {
    return kOpFormat_DEFAULT;
  }
  if (data_type != kNumberTypeFloat16) {
    return kOpFormat_DEFAULT;
  }
  auto dim_0 = shapes[0].size();
  auto dim_1 = shapes[1].size();
  if (dim_0 < special_bmm_size || dim_1 < special_bmm_size) {
    return kOpFormat_DEFAULT;
  }
  auto is_align = [&shapes, dim_0, dim_1]() {
    return (!(static_cast<uint64_t>(shapes[0][dim_0 - 1]) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(shapes[0][dim_0 - 2]) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(shapes[1][dim_1 - 1]) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(shapes[1][dim_1 - 2]) & 0x0000000F));
  };
  if (is_align()) {
    return kOpFormat_DEFAULT;
  }
  return kOpFormat_FRAC_NZ;
}

REGISTER_ACL_OP(BatchMatMul).OutputSelector(&CheckBMMNdSupported);
}  // namespace  mindspore::device::ascend
