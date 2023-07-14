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

#include "tools/converter/adapter/acl/mapper/lp_normalization_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops/op_utils.h"
#include "ops/lp_norm.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNameLpNormInputNum = 1;
}  // namespace
STATUS LpNormalizationMapper::Mapper(const CNodePtr &cnode) {
  /*
   * input1                   input
   *   |                       |
   *   |                       |
   *   |                      / \
   * LpNormalization ===>    /   \
   *   |                    |  LpNorm
   *   |                     \   /
   *   |                      Div
   * output                    |
   *                        output
   */
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  auto origin_input = cnode->inputs()[kNameLpNormInputNum];
  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);
  ops::LpNorm lp_norm_op;
  auto axes_ptr = src_prim->GetAttr(ops::kAxis);
  if (axes_ptr != nullptr) {
    int64_t axis_val = GetValue<int64_t>(axes_ptr);
    std::vector<int64_t> axes_vec;
    axes_vec.push_back(axis_val);
    lp_norm_op.set_axis(axes_vec);
  }
  auto p_ptr = src_prim->GetAttr(ops::kP);
  if (p_ptr != nullptr) {
    int64_t p = GetValue<int64_t>(p_ptr);
    lp_norm_op.set_p(p);
  }
  lp_norm_op.set_epsilon(0);
  lp_norm_op.set_keep_dims(true);
  PrimitivePtr dst_prim = lp_norm_op.GetPrim();
  CHECK_NULL_RETURN(dst_prim);
  value_node->set_value(dst_prim);
  auto graph_manager = func_graph->manager();
  if (graph_manager == nullptr) {
    MS_LOG(ERROR) << "Failed to get func graph manager from cnode " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto new_lpnorm_node =
    NewCNode(cnode, dst_prim, {origin_input}, cnode->abstract()->Clone(), cnode->fullname_with_scope() + "_LpNorm");
  if (new_lpnorm_node == nullptr) {
    MS_LOG(ERROR) << "Failed to create LpNorm node for node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto new_div_node = NewCNode(cnode, prim::kPrimDiv, {origin_input, new_lpnorm_node}, cnode->abstract()->Clone(),
                               cnode->fullname_with_scope() + "_Div");
  if (new_div_node == nullptr) {
    MS_LOG(ERROR) << "Failed to create Div node for node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (!graph_manager->Replace(cnode, new_div_node)) {
    MS_LOG(ERROR) << "Failed to replace LpNorm, cnode " << cnode->fullname_with_scope() << ", input size "
                  << cnode->size();
    return RET_ERROR;
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameLpNormalization, LpNormalizationMapper)
}  // namespace lite
}  // namespace mindspore
