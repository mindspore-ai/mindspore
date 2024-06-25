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
#include "plugin/device/ascend/optimizer/ir_fusion/add_rms_norm_fusion.h"

#include <vector>
#include <string>

#include "mindspore/core/utils/ms_context.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace {
kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(CNodePtr node) {
  std::vector<std::string> inputs_format;
  std::vector<std::string> outputs_format;
  std::vector<TypeId> inputs_type;
  std::vector<TypeId> outputs_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    inputs_type.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, input_index));
    inputs_format.push_back(kOpFormat_DEFAULT);
  }
  size_t output_num = AnfAlgo::GetOutputElementNum(node);
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
}  // namespace

const BaseRef AddRmsNormFusion::DefinePattern() const {
  VectorRef add_rms_norm = VectorRef({prim::kPrimRmsNorm, VectorRef({prim::kPrimAdd, x1_, x2_}), gamma_, eps_});
  return add_rms_norm;
}

const AnfNodePtr AddRmsNormFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                           const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  const std::string fusion_op_name = "AddRmsNorm";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_add_rmsnorm =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_add_rmsnorm) {
    return nullptr;
  }

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);

  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(tensor_add);

  auto shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 0);
  auto shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 1);
  if (shape1 != shape2) {
    return nullptr;
  }

  std::vector<TypeId> add_result_types;
  std::vector<BaseShapePtr> add_result_shapes;
  add_result_types.push_back(common::AnfAlgo::GetOutputInferDataType(tensor_add, 0));
  add_result_shapes.push_back(AnfAlgo::GetOutputDetailShape(tensor_add, 0));

  auto prim = std::make_shared<Primitive>("AddRmsNorm");
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, x2, gamma, eps};
  auto add_rms_norm = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add_rms_norm);

  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  size_t output_num = AnfAlgo::GetOutputElementNum(node);
  for (size_t i = 0; i < output_num; i++) {
    types.push_back(common::AnfAlgo::GetOutputInferDataType(node, i));
    shapes.push_back(AnfAlgo::GetOutputDetailShape(node, i));
  }

  types.push_back(add_result_types[0]);
  shapes.push_back(add_result_shapes[0]);

  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, add_rms_norm.get());
  add_rms_norm->set_scope(node->scope());

  auto build_info = GenerateKernelBuildInfo(add_rms_norm);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_rms_norm.get());

  FuncGraphManagerPtr manager = graph->manager();

  auto prim_getitem = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> add_result_inputs = {NewValueNode(prim_getitem), add_rms_norm,
                                               NewValueNode(static_cast<int64_t>(2))};
  auto add_result = graph->NewCNode(add_result_inputs);

  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, add_result.get());
  add_result->set_scope(tensor_add->scope());

  build_info = GenerateKernelBuildInfo(add_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_result.get());

  (void)manager->Replace(tensor_add, add_result);

  return add_rms_norm;
}
}  // namespace opt
}  // namespace mindspore
