/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "pre_activate/ascend/ir_fusion/mul_add_fusion.h"
#include <vector>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include "session/anf_runtime_algorithm.h"
#include "optimizer/opt.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef MulAddFusion::DefinePattern() const {
  VarPtr mul_x_ = std::make_shared<Var>();
  VarPtr mul_y_ = std::make_shared<Var>();
  VarPtr add_y_ = std::make_shared<Var>();

  VectorRef mul({prim::kPrimMul, mul_x_, mul_y_});
  VectorRef add({prim::kPrimTensorAdd, mul, add_y_});
  return add;
}

const AnfNodePtr MulAddFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const {
  if (graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  auto add = node->cast<CNodePtr>();
  if (add == nullptr || add->inputs().size() != kAddInputNum) {
    return nullptr;
  }
  auto mul_anf = add->input(1);
  if (mul_anf == nullptr) {
    return nullptr;
  }
  auto mul = mul_anf->cast<CNodePtr>();
  if (mul == nullptr || mul->inputs().size() != kMulInputNum) {
    return nullptr;
  }
  if (IsUsedByOthers(graph, mul)) {
    MS_LOG(DEBUG) << "Mul is used by more then two nodes, cannot fuse";
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(kFusedMulAddOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), mul->input(1), mul->input(2), add->input(2)};
  auto fusion_node = graph->NewCNode(inputs);
  fusion_node->set_scope(add->scope());
  fusion_node->set_abstract(add->abstract());
  return fusion_node;
}
}  // namespace opt
}  // namespace mindspore
