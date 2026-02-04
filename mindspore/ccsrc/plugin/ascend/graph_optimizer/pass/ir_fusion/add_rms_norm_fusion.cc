/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion/add_rms_norm_fusion.h"

#include <vector>
#include <string>

#include "utils/ms_context.h"
#include "primitive/nn_optimizer_ops.h"
#include "primitive/nn_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto fusion_op_name = "AddRmsNorm";
}  // namespace

std::vector<std::string> AddRmsNormFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimRmsNorm->name(), prim::kPrimAdd->name()};
  return ret;
}

const BaseRef AddRmsNormFusion::DefinePattern() const {
  VectorRef add_rms_norm = VectorRef({prim::kPrimRmsNorm, VectorRef({prim::kPrimAdd, x1_, x2_}), gamma_, eps_});
  return add_rms_norm;
}

const AnfNodePtr AddRmsNormFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                           const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->IsEnableInferBoost()) {
#ifndef ENABLE_INTERNAL_KERNELS
    return nullptr;
#else
    auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
    bool enable_add_rmsnorm =
      (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
    if (!enable_add_rmsnorm) {
      return nullptr;
    }
#endif  // ENABLE_INTERNAL_KERNELS
  } else {
    // aclnnAddRmsNorm can not support different input types
    auto x_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
    auto gamma_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 1);
    if (x_dtype != gamma_dtype) {
      return nullptr;
    }
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

  auto prim = std::make_shared<Primitive>(fusion_op_name);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, x2, gamma, eps};
  auto add_rms_norm = NewCNode(inputs, graph);
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
  auto add_result = NewCNode(add_result_inputs, graph);

  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, add_result.get());
  add_result->set_scope(tensor_add->scope());

  build_info = GenerateKernelBuildInfo(add_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_result.get());

  (void)manager->Replace(tensor_add, add_result);

  return add_rms_norm;
}
}  // namespace opt
}  // namespace mindspore
