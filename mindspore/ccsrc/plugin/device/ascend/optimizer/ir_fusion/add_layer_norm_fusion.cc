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

#ifndef ENABLE_INTERNAL_KERNELS
#include "plugin/device/ascend/optimizer/ir_fusion/add_layer_norm_fusion.h"

#include <vector>
#include <string>

#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
const BaseRef AddLayernormFusion::DefinePattern() const {
  VectorRef add_layer_norm = VectorRef({prim::kPrimLayerNorm, VectorRef({prim::kPrimAdd, x1_, x2_}), gamma_, beta_,
                                        begin_norm_axis_, begin_params_axis_, eps_});
  return add_layer_norm;
}

const AnfNodePtr AddLayernormFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore

#else
#include "plugin/device/ascend/optimizer/ir_fusion/add_layer_norm_fusion.h"

#include <vector>
#include <string>

#include "mindspore/core/utils/ms_context.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace {
std::vector<std::string> SplitString(const std::string &str, char delim) {
  std::stringstream ss(str);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) {
      elems.emplace_back(item);
    }
  }
  return elems;
}
}  // namespace

const BaseRef AddLayernormFusion::DefinePattern() const {
  VectorRef add_layer_norm = VectorRef({prim::kPrimLayerNorm, VectorRef({prim::kPrimAdd, x1_, x2_}), gamma_, beta_,
                                        begin_norm_axis_, begin_params_axis_, eps_});
  return add_layer_norm;
}

const AnfNodePtr AddLayernormFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  const std::string fusion_op_name = "AddLayerNorm";
  auto &enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_add_layernorm =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_add_layernorm) {
    return nullptr;
  }
  std::vector<std::string> disable_kernel_list = SplitString(common::GetEnv("MS_DISABLE_INTERNAL_KERNELS_LIST"), ',');
  bool enable_aclnn_kernel = std::any_of(disable_kernel_list.begin(), disable_kernel_list.end(),
                                         [&fusion_op_name](const std::string &name) { return name == fusion_op_name; });

  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto beta = utils::cast<AnfNodePtr>((*equiv)[beta_]);
  auto begin_norm_axis = utils::cast<AnfNodePtr>((*equiv)[begin_norm_axis_]);
  auto begin_params_axis = utils::cast<AnfNodePtr>((*equiv)[begin_params_axis_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);

  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(tensor_add);

  auto shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 0);
  auto shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 1);
  if (shape1 != shape2) {
    return nullptr;
  }

  auto x_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  auto gamma_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 1);
  auto beta_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 2);
  if (x_dtype != gamma_dtype || x_dtype != beta_dtype) {
    return nullptr;
  }

  std::vector<TypeId> add_result_types;
  std::vector<BaseShapePtr> add_result_shapes;
  add_result_types.push_back(common::AnfAlgo::GetOutputInferDataType(tensor_add, 0));
  add_result_shapes.push_back(AnfAlgo::GetOutputDetailShape(tensor_add, 0));

  std::vector<AnfNodePtr> inputs;
  if (enable_aclnn_kernel) {
    auto prim = std::make_shared<Primitive>("AddLayerNormV2");
    auto additional_output = CreateValueNodeWithKernelInfo(graph, MakeValue<bool>(false));
    MS_EXCEPTION_IF_NULL(prim);
    MS_EXCEPTION_IF_NULL(additional_output);
    inputs = {NewValueNode(prim), x1, x2, gamma, beta, eps, additional_output};
  } else {
    auto prim = std::make_shared<Primitive>("AddLayerNorm");
    MS_EXCEPTION_IF_NULL(prim);
    inputs = {NewValueNode(prim), x1, x2, gamma, beta, begin_norm_axis, begin_params_axis, eps};
  }

  auto add_layernorm = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add_layernorm);

  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  size_t output_num = AnfAlgo::GetOutputElementNum(node);
  for (size_t i = 0; i < output_num; i++) {
    types.push_back(common::AnfAlgo::GetOutputInferDataType(node, i));
    shapes.push_back(AnfAlgo::GetOutputDetailShape(node, i));
  }
  if (enable_aclnn_kernel) {
    types[kIndex1] = kNumberTypeFloat32;
    types[kIndex2] = kNumberTypeFloat32;
  }

  types.push_back(add_result_types[0]);
  shapes.push_back(add_result_shapes[0]);

  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, add_layernorm.get());
  add_layernorm->set_scope(node->scope());

  auto build_info = GenerateKernelBuildInfo(add_layernorm);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_layernorm.get());

  FuncGraphManagerPtr manager = graph->manager();

  auto prim_getitem = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> add_result_inputs = {NewValueNode(prim_getitem), add_layernorm,
                                               NewValueNode(static_cast<int64_t>(3))};
  auto add_result = graph->NewCNode(add_result_inputs);

  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, add_result.get());
  add_result->set_scope(tensor_add->scope());

  build_info = GenerateKernelBuildInfo(add_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_result.get());

  (void)manager->Replace(tensor_add, add_result);

  return add_layernorm;
}
}  // namespace opt
}  // namespace mindspore
#endif
