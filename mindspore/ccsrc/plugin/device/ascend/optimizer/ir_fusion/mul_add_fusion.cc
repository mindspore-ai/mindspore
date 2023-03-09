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
#include "plugin/device/ascend/optimizer/ir_fusion/mul_add_fusion.h"
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/optimizer/opt.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
bool GetMul(const FuncGraphPtr &graph, const CNodePtr &add, CNodePtr *mul, size_t *mul_index) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(add);

  for (size_t index = 1; index < add->size(); ++index) {
    auto input = add->input(index);
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<CNode>()) {
      auto cnode = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (common::AnfAlgo::GetCNodeName(cnode) == prim::kPrimMul->name()) {
        if (!opt::IsUsedByOthers(graph, cnode)) {
          auto full_name = cnode->fullname_with_scope();
          // exclude lamb and adam, and only work in bert
          if (std::string::npos != full_name.find("adam") || std::string::npos != full_name.find("lamb") ||
              std::string::npos == full_name.find("bert")) {
            MS_LOG(INFO) << "Mul is in adam or lamb or not a bert network, quit fusion";
            return false;
          }

          *mul = cnode;
          *mul_index = index;
          return true;
        }
      }
    }
  }
  return false;
}
}  // namespace
const BaseRef MulAddFusion::DefinePattern() const {
  VarPtr x = std::make_shared<Var>();
  VarPtr y = std::make_shared<Var>();
  VectorRef pattern({prim::kPrimAdd, x, y});
  return pattern;
}

const AnfNodePtr MulAddFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  if (graph == nullptr || node == nullptr) {
    return nullptr;
  }
  auto add = node->cast<CNodePtr>();
  if (add == nullptr || common::AnfAlgo::GetInputTensorNum(add) != kAddInputTensorNum) {
    return nullptr;
  }

  CNodePtr mul = nullptr;
  size_t mul_index = 0;
  if (!GetMul(graph, add, &mul, &mul_index) || mul == nullptr || mul_index == 0) {
    MS_LOG(DEBUG) << "Cannot find used-by-only-one-op Mul in Add's inputs";
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(kFusedMulAddOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
  for (size_t index = 1; index < mul->size(); ++index) {
    inputs.push_back(mul->input(index));
  }
  auto another_input_node = add->input(add->size() - mul_index);
  if (another_input_node->isa<CNode>() &&
      common::AnfAlgo::GetCNodeName(another_input_node) == prim::kPrimTupleGetItem->name()) {
    MS_LOG(INFO) << "Add's another input node has multiple outputs, do not fuse";
    return nullptr;
  }
  inputs.push_back(another_input_node);
  auto fusion_node = NewCNode(inputs, graph);
  fusion_node->set_scope(add->scope());
  fusion_node->set_abstract(add->abstract());
  return fusion_node;
}
}  // namespace opt
}  // namespace mindspore
