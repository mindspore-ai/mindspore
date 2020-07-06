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
#include "pre_activate/gpu/adam_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(CNodePtr node) {
  std::vector<std::string> inputs_format;
  std::vector<std::string> outputs_format;
  std::vector<TypeId> inputs_type;
  std::vector<TypeId> outputs_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;

  for (size_t input_index = 0; input_index < AnfAlgo::GetInputTensorNum(node); ++input_index) {
    inputs_type.push_back(AnfAlgo::GetPrevNodeOutputInferDataType(node, input_index));
    inputs_format.push_back(kOpFormat_DEFAULT);
  }
  for (size_t output_index = 0; output_index < AnfAlgo::GetOutputTensorNum(node); ++output_index) {
    outputs_type.push_back(AnfAlgo::GetOutputInferDataType(node, output_index));
    outputs_format.push_back(kOpFormat_DEFAULT);
  }
  builder.SetInputsDeviceType(inputs_type);
  builder.SetInputsFormat(inputs_format);
  builder.SetOutputsDeviceType(outputs_type);
  builder.SetOutputsFormat(outputs_format);
  return builder.Build();
}
}  // namespace

const BaseRef AdamFusion::DefinePattern() const {
  VectorRef next_m = VectorRef({prim::kPrimTensorAdd, VectorRef({prim::kPrimMul, beta1_, m_}),
                                VectorRef({prim::kPrimMul, one_sub_beta1_, gradient_})});
  VectorRef next_v =
    VectorRef({prim::kPrimTensorAdd, VectorRef({prim::kPrimMul, beta2_, v_}),
               VectorRef({prim::kPrimMul, one_sub_beta2_, VectorRef({prim::kPrimSquare, gradient_})})});
  VectorRef update = VectorRef(
    {prim::kPrimRealDiv, next_m, VectorRef({prim::kPrimTensorAdd, eps_, VectorRef({prim::kPrimSqrt, next_v})})});
  VectorRef update_with_lr = VectorRef({prim::kPrimMul, lr_, update});
  VectorRef next_param = VectorRef({prim::kPrimSub, param_, update_with_lr});
  VectorRef depend1 = VectorRef({prim::kPrimDepend, next_v, VectorRef({prim::kPrimAssign, param_, next_param})});
  VectorRef depend2 = VectorRef({prim::kPrimDepend, depend1, VectorRef({prim::kPrimAssign, m_, next_m})});
  VectorRef depend3 = VectorRef({prim::kPrimDepend, depend2, VectorRef({prim::kPrimAssign, v_, depend2})});
  return depend3;
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

  auto prim = std::make_shared<Primitive>(kFusedAdamName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {
    NewValueNode(prim), beta1_input, one_sub_beta1_input, beta2_input, one_sub_beta2_input,
    eps_input,          lr_input,    param_input,         m_input,     v_input,
    gradient_input};
  auto adam = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(adam);
  auto types = {AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(node, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, adam.get());
  adam->set_scope(node->scope());

  auto build_info = GenerateKernelBuildInfo(adam);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, adam.get());
  return adam;
}
}  // namespace opt
}  // namespace mindspore
