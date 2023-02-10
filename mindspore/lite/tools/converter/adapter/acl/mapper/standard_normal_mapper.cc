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

#include "tools/converter/adapter/acl/mapper/standard_normal_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"
#include "tools/converter/adapter/acl/common/utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNameInputNum = 2;
constexpr size_t kNumFlagThree = 3;
}  // namespace

STATUS StandardNormalMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  if (cnode->size() != kNameInputNum) {
    MS_LOG(ERROR) << "Input size of StandardNormal must be " << (kNameInputNum - 1)
                  << " real size: " << (cnode->size() - 1);
    return lite::RET_ERROR;
  }

  ops::StandardNormal std_norm;
  auto dst_prim = std_norm.GetPrim();
  MSLITE_CHECK_PTR(dst_prim);
  dst_prim->AddAttr("dtype", TypeIdToType(acl::GetTypeFromNode(cnode)));
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameStandardNormal, StandardNormalMapper)
}  // namespace lite
}  // namespace mindspore
