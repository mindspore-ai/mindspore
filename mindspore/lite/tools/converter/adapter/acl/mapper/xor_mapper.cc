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

#include "tools/converter/adapter/acl/mapper/xor_mapper.h"
#include <memory>
#include "src/common/log_util.h"
#include "ops/op_utils.h"
#include "ops/bitwisexor.h"
#include "mindspore/core/ops/array_ops.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kInput1Idx = 1;
constexpr size_t kInput2Idx = 2;
}  // namespace
STATUS XorMapper::Mapper(const CNodePtr &cnode) {
  /*
   * input1(bool)  input2(bool)   input1(bool)          input2(bool)
   * \              /                  \                    /
   *  \            /               Cast1(bool to int)  Cast2(bool to int)
   *   \          /                        \            /
   *    LogicalXor          ===>             BitWiseXor
   *       |                                    |
   *       |                            Cast3(int to bool)
   *       |                                    |
   * output(bool)                           output(bool)
   */
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return RET_ERROR;
  }
  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to get func graph from cnode " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  ops::BitwiseXor bitwiseXor_op;
  auto dst_prim = bitwiseXor_op.GetPrim();
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  auto input1 = cnode->inputs()[kInput1Idx];
  auto input2 = cnode->inputs()[kInput2Idx];
  auto cast_node_1 = NewCNode(cnode, prim::kPrimCast, {input1, NewValueNode(TypeIdToType(kNumberTypeInt8))},
                              cnode->abstract()->Clone(), cnode->fullname_with_scope() + "_1_Cast");
  if (cast_node_1 == nullptr) {
    MS_LOG(ERROR) << "Failed to create Cast node for node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto cast_node_2 = NewCNode(cnode, prim::kPrimCast, {input2, NewValueNode(TypeIdToType(kNumberTypeInt8))},
                              cnode->abstract()->Clone(), cnode->fullname_with_scope() + "_2_Cast");
  if (cast_node_2 == nullptr) {
    MS_LOG(ERROR) << "Failed to create Cast node for node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  value_node->set_value(dst_prim);
  auto graph_manager = func_graph->manager();
  if (graph_manager == nullptr) {
    MS_LOG(ERROR) << "Failed to get func graph manager from cnode " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto new_bitwisexor_node = NewCNode(cnode, dst_prim, {cast_node_1, cast_node_2}, cnode->abstract()->Clone(),
                                      cnode->fullname_with_scope() + "_BitwiseXor");
  if (new_bitwisexor_node == nullptr) {
    MS_LOG(ERROR) << "Failed to create BitwiseXor node for node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto cast_node_3 =
    NewCNode(cnode, prim::kPrimCast, {new_bitwisexor_node, NewValueNode(TypeIdToType(kNumberTypeBool))},
             cnode->abstract()->Clone(), cnode->fullname_with_scope() + "_3_Cast");
  if (cast_node_3 == nullptr) {
    MS_LOG(ERROR) << "Failed to create Cast node for node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (!graph_manager->Replace(cnode, cast_node_3)) {
    MS_LOG(ERROR) << "Failed to replace Cast node, cnode " << cnode->fullname_with_scope() << ", input size "
                  << cnode->size();
    return RET_ERROR;
  }
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameLogicalXor, XorMapper)
}  // namespace lite
}  // namespace mindspore
