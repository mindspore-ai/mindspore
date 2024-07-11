/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/onehot_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "include/registry/converter_context.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
STATUS GetAttrAxis(const AnfNodePtr &cnode, int64_t *result_axis) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_NULL_PTR);

  auto onehot_node = ops::GetOperator<ops::OneHot>(cnode->cast<CNodePtr>()->input(0));
  MS_CHECK_TRUE_RET(onehot_node != nullptr, RET_ERROR);

  auto prim = onehot_node->GetPrim();
  MS_CHECK_TRUE_RET(prim != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kAxis) != nullptr, RET_ERROR);

  *result_axis = GetValue<int64_t>(prim->GetAttr(ops::kAxis));
  return RET_OK;
}

constexpr int kOneHotInputNum = 5;

STATUS AddAttrAxisToLastInput(const CNodePtr &cnode, int64_t axis) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_NULL_PTR, "onthot cnode is nullptr.");
  MS_CHECK_TRUE_RET(cnode->inputs().size() == kOneHotInputNum, RET_NULL_PTR);

  auto func_graph = cnode->func_graph();
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_NULL_PTR, "func_graph is nullptr.");

  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_NULL_PTR, "manager is nullptr.");

  auto axis_node = NewValueNode(MakeValue<int64_t>(axis));
  MS_CHECK_TRUE_MSG(axis_node != nullptr, RET_NULL_PTR, "axis node is nullptr.");

  manager->AddEdge(cnode, axis_node);

  return RET_OK;
}
}  // namespace

// in ascend graph ir, OneHot' attribute 'axis' is of input_to_attr type, but in
// lite's OneHot, it's an attribute. so we need to add it as the last input here.
STATUS OneHotMapper::Mapper(const CNodePtr &cnode) {
  int64_t axis = 0;

  auto status = GetAttrAxis(cnode, &axis);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get attribute 'axis' failed, ret: " << status;
    return status;
  }
  MS_LOG(DEBUG) << "the onehot axis value is " << axis;

  status = AddAttrAxisToLastInput(cnode, axis);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "add 'axis' of op onehot as the last input failed, ret: " << status;
    return status;
  }

  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameOneHot, OneHotMapper)
}  // namespace lite
}  // namespace mindspore
