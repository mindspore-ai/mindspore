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

#include "tools/optimizer/fusion/gelu_fusion.h"
#include <memory>
#include <string>
#include "ops/fusion/activation.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
bool GeLUFusion::Init() const {
  input_ = std::make_shared<Var>();
  return input_ != nullptr;
}

CNodePtr GeLUFusion::CreateGeLUNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                    const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  auto gelu_prim = std::make_shared<ops::Activation>();
  MS_CHECK_TRUE_RET(gelu_prim != nullptr, nullptr);
  gelu_prim->set_activation_type(mindspore::GELU);
  gelu_prim->set_approximate(approximate_);
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_ASSERT(input_node != nullptr);
  auto gelu_cnode = func_graph->NewCNode(gelu_prim, {input_node});
  MS_CHECK_TRUE_RET(gelu_cnode != nullptr, nullptr);
  gelu_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_gelu");
  if (node->abstract() != nullptr) {
    gelu_cnode->set_abstract(node->abstract()->Clone());
  }
  return gelu_cnode;
}

const float GeLUFusion::GetParameterValue(const EquivPtr &equiv, const VarPtr &input) const {
  MS_ASSERT(equiv != nullptr && input != nullptr);
  const float value = -1;
  auto node = utils::cast<AnfNodePtr>((*equiv)[input]);
  if (node == nullptr || !utils::isa<ParameterPtr>(node)) {
    return value;
  }
  auto parameter_node = node->cast<ParameterPtr>();
  if (!parameter_node->has_default() || parameter_node->default_param() == nullptr) {
    return value;
  }
  auto param_value_lite = parameter_node->default_param()->cast<tensor::TensorPtr>();
  if (param_value_lite == nullptr) {
    return value;
  }
  if (param_value_lite->data_type() != kNumberTypeFloat32 && param_value_lite->data_type() != kNumberTypeFloat) {
    return value;
  }
  if (param_value_lite->Size() != sizeof(float)) {
    return value;
  }
  return *static_cast<float *>(param_value_lite->data_c());
}

const AnfNodePtr GeLUFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                     const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    return nullptr;
  }
  if (!CheckPattern(equiv)) {
    return nullptr;
  }
  auto cnode = CreateGeLUNode(func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "new gelu node failed.";
    return nullptr;
  }
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
