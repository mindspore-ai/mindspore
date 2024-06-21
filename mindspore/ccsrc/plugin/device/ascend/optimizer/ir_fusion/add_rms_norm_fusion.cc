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
#include "include/backend/distributed/collective/collective_manager.h"

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
bool IsC(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    AnfNodePtr in = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    return in->isa<ValueNode>();
  }
  return false;
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

const BaseRef RmsNormQuantFusion::DefinePattern() const {
  VarPtr index0 = std::make_shared<CondVar>(IsC);
  VectorRef rms_norm = VectorRef({prim::kPrimRmsNorm, x1_, gamma_, eps_});
  VectorRef tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, rms_norm, index0});
  VarPtr x0 = std::make_shared<CondVar>(IsC);
  VarPtr x1 = std::make_shared<CondVar>(IsC);
  VarPtr x2 = std::make_shared<CondVar>(IsC);
  VectorRef rms_norm_quant = VectorRef({prim::kPrimQuantV2, tuple_get_item_0, scale_, offset_, x0, x1, x2});
  return rms_norm_quant;
}

const AnfNodePtr RmsNormQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  const std::string fusion_op_name = "RmsNormQuant";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_add_rmsnorm =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_add_rmsnorm) {
    return nullptr;
  }

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto beta = NewValueNode(kNone);
  beta->set_abstract(kNone->ToAbstract());
  auto scale = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  auto offset = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  auto epsilon_value = GetValueNode(utils::cast<AnfNodePtr>((*equiv)[eps_]));
  MS_EXCEPTION_IF_NULL(epsilon_value);

  auto tuple_get_item_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tuple_get_item_node), 0);
  MS_EXCEPTION_IF_NULL(rms_norm_node);
  auto prim = std::make_shared<Primitive>("RmsNormQuant");
  prim->set_attr("epsilon", epsilon_value);

  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, gamma, gamma, scale, offset};
  auto rms_norm_quant = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(rms_norm_quant);

  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  size_t output_num = AnfAlgo::GetOutputElementNum(node);
  for (size_t i = 0; i < output_num; i++) {
    types.push_back(common::AnfAlgo::GetOutputInferDataType(node, i));
    shapes.push_back(AnfAlgo::GetOutputDetailShape(node, i));
  }

  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, rms_norm_quant.get());
  rms_norm_quant->set_scope(node->scope());

  auto build_info = GenerateKernelBuildInfo(rms_norm_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, rms_norm_quant.get());

  return rms_norm_quant;
}

const BaseRef AddRmsNormQuantFusion::DefinePattern() const {
  VectorRef add_rms_norm = VectorRef({prim::kPrimRmsNorm, VectorRef({prim::kPrimAdd, x1_, x2_}), gamma_, eps_});
  VarPtr index0 = std::make_shared<CondVar>(IsC);
  VectorRef tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, add_rms_norm, index0});
  sqrt_mode_ = std::make_shared<CondVar>(IsC);
  rounding_mode_ = std::make_shared<CondVar>(IsC);
  dst_type_ = std::make_shared<CondVar>(IsC);
  VectorRef add_rms_norm_quant =
    VectorRef({prim::kPrimQuantV2, tuple_get_item_0, scale_, offset_, sqrt_mode_, rounding_mode_, dst_type_});
  return add_rms_norm_quant;
}

const AnfNodePtr AddRmsNormQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  const std::string fusion_op_name = "AddRmsNormQuantV2";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_add_rmsnorm =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_add_rmsnorm) {
    return nullptr;
  }
  auto tuple_get_item_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tuple_get_item_node), 0);
  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_node), 0);
  auto shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 0);
  auto shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 1);
  if (shape1 != shape2) {
    return nullptr;
  }

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto scale = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  auto offset = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);
  auto sqrt_mode = GetValueNode(utils::cast<AnfNodePtr>((*equiv)[sqrt_mode_]));
  auto rounding_mode = GetValueNode(utils::cast<AnfNodePtr>((*equiv)[rounding_mode_]));
  auto dst_type = GetValueNode(utils::cast<AnfNodePtr>((*equiv)[dst_type_]));

  auto tensor_rms_norm = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(tensor_rms_norm);

  auto prim = std::make_shared<Primitive>("AddRmsNormQuantV2");
  prim->set_attr("sqrt_mode", sqrt_mode);
  prim->set_attr("rounding_mode", rounding_mode);
  prim->set_attr("dst_type", dst_type);

  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, x2, gamma, scale, offset, eps};
  auto add_rms_norm_quant = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add_rms_norm_quant);

  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  std::vector<TypeId> add_result_types;
  std::vector<BaseShapePtr> add_result_shapes;
  std::vector<TypeId> quant_result_types;
  std::vector<BaseShapePtr> quant_result_shapes;
  size_t output_num = AnfAlgo::GetOutputElementNum(node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << node->cast<CNodePtr>()->fullname_with_scope() << " output_num " << output_num << " != 1.";
  }
  auto tensor_quant_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  auto tensor_quant_shape = AnfAlgo::GetOutputDetailShape(node, 0);
  auto tensor_add_type = common::AnfAlgo::GetOutputInferDataType(tensor_add, 0);
  auto tensor_add_shape = AnfAlgo::GetOutputDetailShape(tensor_add, 0);

  for (size_t i = 0; i < output_num + 1; i++) {
    types.push_back(tensor_quant_type);
    shapes.push_back(tensor_quant_shape);
  }
  types.push_back(tensor_add_type);
  shapes.push_back(tensor_add_shape);
  quant_result_types.push_back(tensor_quant_type);
  quant_result_shapes.push_back(tensor_quant_shape);
  add_result_types.push_back(tensor_add_type);
  add_result_shapes.push_back(tensor_add_shape);

  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, add_rms_norm_quant.get());
  add_rms_norm_quant->set_scope(node->scope());

  auto build_info = GenerateKernelBuildInfo(add_rms_norm_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_rms_norm_quant.get());

  FuncGraphManagerPtr manager = graph->manager();
  auto prim_getitem_2 = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> add_result_inputs = {NewValueNode(prim_getitem_2), add_rms_norm_quant,
                                               NewValueNode(static_cast<int64_t>(2))};
  auto add_result = graph->NewCNode(add_result_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, add_result.get());
  add_result->set_scope(tensor_add->scope());
  build_info = GenerateKernelBuildInfo(add_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_result.get());
  (void)manager->Replace(tensor_add, add_result);

  auto prim_getitem_0 = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> quant_result_inputs = {NewValueNode(prim_getitem_0), add_rms_norm_quant,
                                                 NewValueNode(static_cast<int64_t>(0))};
  auto quant_result = graph->NewCNode(quant_result_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(quant_result_types, quant_result_shapes, quant_result.get());
  quant_result->set_scope(node->scope());
  build_info = GenerateKernelBuildInfo(quant_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, quant_result.get());

  return quant_result;
}

}  // namespace opt
}  // namespace mindspore
