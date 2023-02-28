/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/mul_addn_fusion.h"
#include <vector>
#include <memory>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/optimizer/opt.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
CNodePtr MulAddNFusion::CreateFusionNode(const FuncGraphPtr &graph, const CNodePtr &mul, const CNodePtr &addn,
                                         const size_t &lossscale_input_index) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mul);
  MS_EXCEPTION_IF_NULL(addn);
  auto prim = std::make_shared<Primitive>(kFusedMulAddNOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
  inputs.push_back(mul->input(kMulInputTensorNum + 1 - lossscale_input_index));
  inputs.push_back(addn->input(kIndex2));
  // scalar input should be 3rd input
  inputs.push_back(mul->input(lossscale_input_index));
  auto fusion_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_scope(addn->scope());
  fusion_node->set_abstract(addn->abstract());
  return fusion_node;
}

const BaseRef MulAddNFusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  VarPtr Z = std::make_shared<Var>();

  VectorRef mul({prim::kPrimMul, X, Z});
  VectorRef addn({prim::kPrimAddN, mul, Y});
  return addn;
}

const AnfNodePtr MulAddNFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                        const EquivPtr &equiv) const {
  if (graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }

  auto addn = node->cast<CNodePtr>();
  if (addn == nullptr) {
    return nullptr;
  }
  if (common::AnfAlgo::IsDynamicShape(addn)) {
    return nullptr;
  }

  auto mul_anf = addn->input(kIndex1);
  if (mul_anf == nullptr) {
    return nullptr;
  }
  auto mul = mul_anf->cast<CNodePtr>();
  if (mul == nullptr || common::AnfAlgo::GetInputTensorNum(mul) != kMulInputTensorNum) {
    return nullptr;
  }
  if (common::AnfAlgo::IsDynamicShape(mul)) {
    return nullptr;
  }
  if (IsUsedByOthers(graph, mul)) {
    MS_LOG(DEBUG) << "Mul is used by more then two nodes, cannot fuse";
    return nullptr;
  }

  size_t lossscale_input_index = 1;
  for (size_t index = 1; index < mul->inputs().size(); ++index) {
    auto input_node = mul->input(index);
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<ValueNode>()) {
      lossscale_input_index = index;
      break;
    }
  }

  auto constant_shape = common::AnfAlgo::GetOutputInferShape(mul->input(lossscale_input_index), 0);
  if (!(constant_shape.size() == 0 || (constant_shape.size() == 1 && constant_shape[0] == 1))) {
    MS_LOG(DEBUG) << "The const input of Mul node must be scalar or shape=(1,), but shape size is "
                  << constant_shape.size() << " and shape[0] is " << constant_shape[0];
    return nullptr;
  }

  return CreateFusionNode(graph, mul, addn, lossscale_input_index);
}
}  // namespace opt
}  // namespace mindspore
