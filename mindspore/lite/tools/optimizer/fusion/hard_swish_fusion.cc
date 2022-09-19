/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/fusion/hard_swish_fusion.h"
#include <vector>
#include "nnacl/op_base.h"
#include "ops/fusion/activation.h"

namespace mindspore {
namespace opt {
namespace {
constexpr float kHSwishAddConst = 3.0;
constexpr float kHSwishDivConst = 6.0;
}  // namespace

const BaseRef HardSwishFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }

  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  VectorRef add_ref = VectorRef({is_add, input_, add_const_});

  auto is_relu6 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_relu6 != nullptr, {});
  VectorRef relu6_ref = VectorRef({is_relu6, add_ref});

  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  VectorRef mul_ref = VectorRef({is_mul, input_, relu6_ref});

  auto is_div = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  MS_CHECK_TRUE_RET(is_div != nullptr, {});
  VectorRef div_ref = VectorRef({is_div, mul_ref, div_const_});

  return div_ref;
}

const AnfNodePtr HardSwishFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "input param is nullptr, do norm fusion failed.";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  auto div_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(div_cnode != nullptr, nullptr);
  if (IsMarkedTrainOp(div_cnode)) {
    return nullptr;
  }
  if (!CheckPattern(func_graph, equiv)) {
    return nullptr;
  }

  // create new node
  auto hard_swish_primitive = std::make_shared<ops::Activation>();
  MS_CHECK_TRUE_RET(hard_swish_primitive != nullptr, nullptr);
  hard_swish_primitive->Init(0, 0, FLT_MAX, mindspore::HSWISH);
  auto hard_swish_primitive_c = hard_swish_primitive->GetPrim();
  MS_CHECK_TRUE_RET(hard_swish_primitive_c != nullptr, nullptr);

  auto value_node = NewValueNode(hard_swish_primitive_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> new_node_inputs = {value_node};
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_CHECK_TRUE_RET(input_node != nullptr, nullptr);
  new_node_inputs.push_back(input_node);
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);

  new_node->set_abstract(div_cnode->abstract()->Clone());
  new_node->set_fullname_with_scope("hard_swish_" + div_cnode->fullname_with_scope());
  return new_node;
}

bool HardSwishFusion::Init() const {
  input_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_ != nullptr, false);
  add_const_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add_const_ != nullptr, false);
  div_const_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(div_const_ != nullptr, false);

  return true;
}

bool HardSwishFusion::CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv) const {
  // add const
  auto add_const_node = utils::cast<AnfNodePtr>((*equiv)[add_const_]);
  MS_CHECK_TRUE_RET(add_const_node != nullptr, false);
  if (!add_const_node->isa<Parameter>()) {
    return false;
  }
  auto add_const_param = add_const_node->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(add_const_param != nullptr, false);
  auto add_const_tensor = add_const_param->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_RET(add_const_tensor != nullptr, false);
  auto add_const_shape = add_const_tensor->shape();
  if (add_const_shape.empty() || (add_const_shape.size() == 1 && add_const_shape[0] == 1)) {
    MS_CHECK_TRUE_RET(add_const_tensor->data_c() != nullptr, false);
    auto const_data = reinterpret_cast<float *>(add_const_tensor->data_c());
    if (const_data[0] != kHSwishAddConst) {
      return false;
    }
  } else {
    return false;
  }

  // div const
  auto div_const_node = utils::cast<AnfNodePtr>((*equiv)[div_const_]);
  MS_CHECK_TRUE_RET(div_const_node != nullptr, false);
  if (!div_const_node->isa<Parameter>()) {
    return false;
  }
  auto div_const_param = div_const_node->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(div_const_param != nullptr, false);
  auto div_const_tensor = div_const_param->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_RET(div_const_tensor != nullptr, false);
  auto div_const_shape = div_const_tensor->shape();
  if (div_const_shape.empty() || (div_const_shape.size() == 1 && div_const_shape[0] == 1)) {
    MS_CHECK_TRUE_RET(div_const_tensor->data_c() != nullptr, false);
    auto const_data = reinterpret_cast<float *>(div_const_tensor->data_c());
    if (const_data[0] != kHSwishDivConst) {
      return false;
    }
  } else {
    return false;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
