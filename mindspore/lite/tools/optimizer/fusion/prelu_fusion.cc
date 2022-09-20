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

#include "tools/optimizer/fusion/prelu_fusion.h"
#include "nnacl/op_base.h"
#include "ops/fusion/prelu_fusion.h"

namespace mindspore {
namespace opt {
bool PReluFusion::Init() const {
  input_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_ != nullptr, false);
  mul_const_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul_const_ != nullptr, false);
  return true;
}

const BaseRef PReluFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }

  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  VectorRef mul_ref = VectorRef({is_mul, mul_const_, input_});

  auto is_maximum = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMaximum>);
  MS_CHECK_TRUE_RET(is_maximum != nullptr, {});
  VectorRef maximum_ref = VectorRef({is_maximum, mul_ref, input_});

  return maximum_ref;
}

const AnfNodePtr PReluFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                      const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "input param is nullptr, do norm fusion failed.";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  auto maximum_cnode = node->cast<CNodePtr>();
  if (IsMarkedTrainOp(maximum_cnode)) {
    return nullptr;
  }
  std::vector<float> slope;
  if (!CheckPattern(func_graph, equiv, &slope)) {
    return nullptr;
  }

  // create new node
  auto prelu_primitive = std::make_shared<ops::PReLUFusion>();
  MS_CHECK_TRUE_RET(prelu_primitive != nullptr, nullptr);
  if (slope.size() == 1) {
    prelu_primitive->Init(true, slope);
  } else if (slope.size() > 1) {
    prelu_primitive->Init(false, slope);
  } else {
    MS_LOG(ERROR) << "slope const is empty.";
    return nullptr;
  }
  auto prelu_primitive_c = prelu_primitive->GetPrim();
  MS_CHECK_TRUE_RET(prelu_primitive_c != nullptr, nullptr);
  auto value_node = NewValueNode(prelu_primitive_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> new_node_inputs = {value_node};
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_CHECK_TRUE_RET(input_node != nullptr, nullptr);
  new_node_inputs.push_back(input_node);

  auto mul_node = node->cast<CNodePtr>()->input(kInputIndexOne);
  MS_CHECK_TRUE_RET(mul_node != nullptr, nullptr);
  auto bias_node = mul_node->cast<CNodePtr>()->input(kInputIndexOne);
  MS_CHECK_TRUE_RET(bias_node != nullptr, nullptr);
  new_node_inputs.push_back(bias_node);
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);

  new_node->set_abstract(maximum_cnode->abstract()->Clone());
  new_node->set_fullname_with_scope("prelu_" + maximum_cnode->fullname_with_scope());
  return new_node;
}

bool PReluFusion::CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv, std::vector<float> *slope) const {
  // mul const
  auto mul_const_cnode = utils::cast<AnfNodePtr>((*equiv)[mul_const_]);
  MS_CHECK_TRUE_RET(mul_const_cnode != nullptr, false);
  if (!mul_const_cnode->isa<Parameter>()) {
    return false;
  }
  auto mul_const_param = mul_const_cnode->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(mul_const_param != nullptr, false);
  auto mul_const_tensor = mul_const_param->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_RET(mul_const_tensor != nullptr, false);
  auto mul_const_shape = mul_const_tensor->shape();
  MS_CHECK_TRUE_RET(mul_const_tensor->data_c() != nullptr, false);
  auto const_data = reinterpret_cast<float *>(mul_const_tensor->data_c());
  if (mul_const_shape.empty() || (mul_const_shape.size() == 1 && mul_const_shape[0] == 1)) {
    slope->push_back(const_data[0]);
  } else {
    for (int offset = 0; offset < mul_const_tensor->ElementsNum(); offset++) {
      slope->push_back(*(const_data + offset));
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
