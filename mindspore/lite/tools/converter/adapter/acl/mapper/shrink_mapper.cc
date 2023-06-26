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

#include "tools/converter/adapter/acl/mapper/shrink_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "mindspore/core/ops/op_name.h"

namespace mindspore {
namespace lite {
const auto kNameShrink = "Shrink";

namespace {
constexpr int64_t kInputMinNum = 1;
}  // namespace

STATUS ShrinkMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  auto dst_prim = std::make_shared<acl::Shrink>();
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  int64_t num = static_cast<int64_t>(cnode->inputs().size());
  if (num < kInputMinNum) {
    MS_LOG(ERROR) << "Input size " << num << " is less than " << kInputMinNum;
    return RET_ERROR;
  }

  auto lambd = src_prim->GetAttr(mindspore::ops::kLambd);
  auto bias = src_prim->GetAttr(mindspore::ops::kBias);
  if (lambd != nullptr) {
    dst_prim->AddAttr("lambd", lambd);
  }
  if (bias != nullptr) {
    dst_prim->AddAttr("bias", bias);
  }

  value_node->set_value(dst_prim);

  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);

  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameShrink, ShrinkMapper)
}  // namespace lite
}  // namespace mindspore
