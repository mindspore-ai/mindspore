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
#include "plugin/device/ascend/optimizer/ir_fusion/inference_swiglu_fusion.h"
#include <vector>
#include "plugin/device/ascend/optimizer/common/gllo_utils.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {

CNodePtr InferenceSwiGLUFusion::CreateSwiGLUNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "start create SwiGLU node";
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  std::string prim_name = "Swiglu";
  auto glu_prim = std::make_shared<Primitive>(prim_name);
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_ASSERT(input_node != nullptr);
  auto axis_node = utils::cast<AnfNodePtr>((*equiv)[axis_]);
  MS_ASSERT(axis_node != nullptr);
  if (!axis_node->isa<ValueNode>()) {
    MS_LOG(DEBUG) << "axis node is not a value node";
    return nullptr;
  }
  std::vector<AnfNodePtr> glu_inputs = {input_node, axis_node};
  auto glu_cnode = func_graph->NewCNode(glu_prim, glu_inputs);
  MS_CHECK_TRUE_RET(glu_cnode != nullptr, nullptr);
  glu_cnode->set_fullname_with_scope(node->fullname_with_scope() + "-SwiGLU");
  if (node->abstract() != nullptr) {
    glu_cnode->set_abstract(node->abstract()->Clone());
  }
  MS_LOG(DEBUG) << "create SwiGLU node success.";
  return glu_cnode;
}

bool InferenceSwiGLUFusion::Init() const {
  input_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_ != nullptr, false);
  split_size_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(split_size_ != nullptr, false);
  axis_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(axis_ != nullptr, false);
  split_prim_ = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSplitWithSize>);
  MS_CHECK_TRUE_RET(split_prim_ != nullptr, false);
  return true;
}

const BaseRef InferenceSwiGLUFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(DEBUG) << "initial member failed.";
    return {};
  }
  VectorRef split_ref({split_prim_, input_, split_size_, axis_});
  auto is_tuple_getitem0 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTupleGetItem>);
  MS_CHECK_TRUE_RET(is_tuple_getitem0 != nullptr, {});
  auto is_seq_var0 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var0 != nullptr, {});
  VectorRef tuple_ref0({is_tuple_getitem0, split_ref, is_seq_var0});
  auto is_tuple_getitem1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTupleGetItem>);
  MS_CHECK_TRUE_RET(is_tuple_getitem1 != nullptr, {});
  auto is_seq_var1 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var1 != nullptr, {});
  VectorRef tuple_ref1({is_tuple_getitem1, split_ref, is_seq_var1});
  auto is_activation = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSiLU>);
  MS_CHECK_TRUE_RET(is_activation != nullptr, {});
  VectorRef sigmoid_ref({is_activation, tuple_ref1});
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMul>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  VectorRef mul_ref({is_mul, tuple_ref0, sigmoid_ref});
  return mul_ref;
}

const AnfNodePtr InferenceSwiGLUFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  constexpr auto kInferenceSwiGLUName = "InferenceSwiGLU";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  auto enable_fusion =
    (std::find(enable_op_list.begin(), enable_op_list.end(), kInferenceSwiGLUName) != enable_op_list.end());
  if (!enable_fusion) {
    return nullptr;
  }

  MS_LOG(DEBUG) << "swiglu_fusion pass";
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }

  auto cnode = CreateSwiGLUNode(func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "create swiglu node failed.";
    return nullptr;
  }
  return cnode;
}

}  // namespace opt
}  // namespace mindspore
