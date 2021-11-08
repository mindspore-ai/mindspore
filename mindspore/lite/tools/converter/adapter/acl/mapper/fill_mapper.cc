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

#include "tools/converter/adapter/acl/mapper/fill_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kInputNum = 3;
constexpr auto kPrimIndex = 0;
constexpr auto kValueIndex = 1;
constexpr auto kDimsIndex = 2;
}  // namespace

STATUS FillMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  auto dst_prim = std::make_shared<acl::FillV1>();
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  if (AdjustInputOrder(cnode) != RET_OK) {
    MS_LOG(ERROR) << "Adjust input order failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS FillMapper::AdjustInputOrder(const CNodePtr &cnode) {
  // original input order: value, dims
  // new order: dims value
  if (cnode->inputs().size() != kInputNum) {
    MS_LOG(ERROR) << "Input num must be " << kInputNum << ",real num " << cnode->inputs().size();
    return lite::RET_ERROR;
  }
  auto inputs = cnode->inputs();
  std::vector<AnfNodePtr> new_inputs = {inputs[kPrimIndex], inputs[kDimsIndex], inputs[kValueIndex]};
  cnode->set_inputs(new_inputs);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameFill, FillMapper)
}  // namespace lite
}  // namespace mindspore
