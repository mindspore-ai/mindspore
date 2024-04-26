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
#include "plugin/device/ascend/optimizer/ir_fission/matmul_add_bias.h"

#include <vector>
#include <memory>

#include "ops/framework_op_name.h"
#include "ops/array_ops.h"
#include "ops/math_op_name.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kMatMulInputSize = 4;
}  // namespace
const AnfNodePtr MatMulAddBias::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = CheckAnfNodeIfCNodeAndInputSize(node, kMatMulInputSize);
  auto none_value = std::make_shared<None>();
  auto none_node = NewValueNode(none_value);
  none_node->set_abstract(none_value->ToAbstract());
  auto new_inputs =
    std::vector<AnfNodePtr>{cnode->input(kIndex0), cnode->input(kIndex1), cnode->input(kIndex2), none_node,
                            cnode->input(kIndex3), cnode->input(kIndex4)};
  auto new_cnode = NewCNode(new_inputs, graph);
  new_cnode->set_scope(node->scope());
  new_cnode->set_abstract(cnode->abstract());
  return new_cnode;
}

const BaseRef MatMulAddBias::DefinePattern() const {
  VarPtr X1 = std::make_shared<Var>();
  VarPtr X2 = std::make_shared<Var>();
  VarPtr X3 = std::make_shared<Var>();
  VarPtr X4 = std::make_shared<Var>();
  return VectorRef({prim::kPrimMatMul, X1, X2, X3, X4});
}
}  // namespace opt
}  // namespace mindspore
