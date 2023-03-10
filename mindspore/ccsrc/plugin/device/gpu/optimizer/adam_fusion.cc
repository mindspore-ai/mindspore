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
#include "plugin/device/gpu/optimizer/adam_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(CNodePtr node) {
  std::vector<std::string> inputs_format;
  std::vector<std::string> outputs_format;
  std::vector<TypeId> inputs_type;
  std::vector<TypeId> outputs_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;

  MS_EXCEPTION_IF_NULL(node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    inputs_type.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, input_index));
    inputs_format.push_back(kOpFormat_DEFAULT);
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(node, output_index));
    outputs_format.push_back(kOpFormat_DEFAULT);
  }
  builder.SetInputsDeviceType(inputs_type);
  builder.SetInputsFormat(inputs_format);
  builder.SetOutputsDeviceType(outputs_type);
  builder.SetOutputsFormat(outputs_format);
  return builder.Build();
}

AnfNodePtr RelpaceOutputEdge(const AnfNodePtr &node, CNodePtr adam, AnfNodePtr u_input) {
  // Replace the parameters of the last UpdateState to maintain
  // the execution order of FusedAdam and the following operators.
  // n represents the operator assign_v in {prim::kPrimDepend, next_param, assign_v}
  const size_t assign_index = 2;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &n = cnode->input(assign_index);
  MS_EXCEPTION_IF_NULL(n);
  const auto &fg = n->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto mgr = fg->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  auto &node_users = mgr->node_users();
  auto iter = node_users.find(n);
  if (iter == node_users.end()) {
    MS_LOG(EXCEPTION) << "Can not find node : " << n->DebugString();
  }

  auto &users = iter->second;
  for (auto &user : users) {
    if (IsPrimitiveCNode(user.first, prim::kPrimUpdateState)) {
      const size_t monad_index = 1;
      const size_t adam_index = 2;
      auto cnode_ptr = (user.first)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode_ptr);
      cnode_ptr->set_input(monad_index, u_input);
      cnode_ptr->set_input(adam_index, adam);
      break;
    }
  }
  return adam;
}
}  // namespace

const BaseRef AdamFusion::DefinePattern() const {
  VectorRef load_param = VectorRef({prim::kPrimLoad, param_, u_});
  VectorRef load_m = VectorRef({prim::kPrimLoad, m_, u_});
  VectorRef load_v = VectorRef({prim::kPrimLoad, v_, u_});

  VectorRef next_m = VectorRef({prim::kPrimAdd, VectorRef({prim::kPrimMul, beta1_, load_m}),
                                VectorRef({prim::kPrimMul, one_sub_beta1_, gradient_})});
  VectorRef next_v =
    VectorRef({prim::kPrimAdd, VectorRef({prim::kPrimMul, beta2_, load_v}),
               VectorRef({prim::kPrimMul, one_sub_beta2_, VectorRef({prim::kPrimSquare, gradient_})})});

  VectorRef update =
    VectorRef({prim::kPrimRealDiv, next_m, VectorRef({prim::kPrimAdd, eps_, VectorRef({prim::kPrimSqrt, next_v})})});
  VectorRef update_with_lr = VectorRef({prim::kPrimMul, lr_, update});
  VectorRef next_param = VectorRef({prim::kPrimSub, load_param, update_with_lr});

  VectorRef tuple_load = VectorRef({prim::kPrimMakeTuple, load_param, load_m, load_v});
  VectorRef next_state = VectorRef({prim::kPrimUpdateState, u_, tuple_load});

  VectorRef assign_param = VectorRef({prim::kPrimAssign, param_, next_param, next_state});
  next_state = VectorRef({prim::kPrimUpdateState, next_state, assign_param});
  next_param = VectorRef({prim::kPrimDepend, next_param, assign_param});

  VectorRef assign_m = VectorRef({prim::kPrimAssign, m_, next_m, next_state});
  next_state = VectorRef({prim::kPrimUpdateState, next_state, assign_m});
  next_param = VectorRef({prim::kPrimDepend, next_param, assign_m});

  VectorRef assign_v = VectorRef({prim::kPrimAssign, v_, next_v, next_state});
  next_param = VectorRef({prim::kPrimDepend, next_param, assign_v});
  return next_param;
}

const AnfNodePtr AdamFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto beta1_input = utils::cast<AnfNodePtr>((*equiv)[beta1_]);
  auto one_sub_beta1_input = utils::cast<AnfNodePtr>((*equiv)[one_sub_beta1_]);
  auto beta2_input = utils::cast<AnfNodePtr>((*equiv)[beta2_]);
  auto one_sub_beta2_input = utils::cast<AnfNodePtr>((*equiv)[one_sub_beta2_]);
  auto eps_input = utils::cast<AnfNodePtr>((*equiv)[eps_]);
  auto lr_input = utils::cast<AnfNodePtr>((*equiv)[lr_]);
  auto param_input = utils::cast<AnfNodePtr>((*equiv)[param_]);
  auto m_input = utils::cast<AnfNodePtr>((*equiv)[m_]);
  auto v_input = utils::cast<AnfNodePtr>((*equiv)[v_]);
  auto gradient_input = utils::cast<AnfNodePtr>((*equiv)[gradient_]);
  auto u_input = utils::cast<AnfNodePtr>((*equiv)[u_]);
  MS_EXCEPTION_IF_NULL(beta1_input);
  MS_EXCEPTION_IF_NULL(one_sub_beta1_input);
  MS_EXCEPTION_IF_NULL(beta2_input);
  MS_EXCEPTION_IF_NULL(one_sub_beta2_input);
  MS_EXCEPTION_IF_NULL(eps_input);
  MS_EXCEPTION_IF_NULL(lr_input);
  MS_EXCEPTION_IF_NULL(param_input);
  MS_EXCEPTION_IF_NULL(m_input);
  MS_EXCEPTION_IF_NULL(v_input);
  MS_EXCEPTION_IF_NULL(gradient_input);
  MS_EXCEPTION_IF_NULL(u_input);

  // Use depend(param, u) to maintain the execution order of FusedAdam and the previous operators.
  auto prim_depend = std::make_shared<Primitive>(prim::kPrimDepend->name());
  MS_EXCEPTION_IF_NULL(prim_depend);
  std::vector<AnfNodePtr> param_inputs = {NewValueNode(prim_depend), param_input, u_input};
  auto param = graph->NewCNode(param_inputs);
  MS_EXCEPTION_IF_NULL(param);
  param->set_abstract(param_input->abstract());

  // Fused into a FusedAdam operator.
  auto prim = std::make_shared<Primitive>(kFusedAdamName);
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_value = NewValueNode(prim);
  std::vector<AnfNodePtr> inputs = {
    prim_value, beta1_input, one_sub_beta1_input, beta2_input, one_sub_beta2_input, eps_input, lr_input, param,
    m_input,    v_input,     gradient_input};
  auto adam = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(adam);
  auto types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(node, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, adam.get());
  adam->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(adam);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, adam.get());
  return RelpaceOutputEdge(node, adam, u_input);
}
}  // namespace opt
}  // namespace mindspore
