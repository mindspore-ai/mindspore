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

#include "plugin/device/ascend/optimizer/ir_fusion/inference_multi_matmul_with_split_fusion.h"

#include <vector>
#include <map>
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_info.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "include/api/data_type.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/utils/ms_context.h"

namespace mindspore {
namespace opt {

static std::map<std::string, ValueNodePtr> weights_with_split_cache_map;

bool InferenceMultiMatmulWithSplitFusion::Run(const FuncGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return false;
  }
  constexpr auto kInferenceMultiMatmulWithSplitOpName = "InferenceMultiMatmulWithSplit";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable = (std::find(enable_op_list.begin(), enable_op_list.end(), kInferenceMultiMatmulWithSplitOpName) !=
                 enable_op_list.end());
  if (!enable) {
    return false;
  }
  bool changed = false;

  auto mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  const auto &node_users_map = mng->node_users();
  auto node_list = TopoSort(graph->output());
  constexpr size_t weight_num_three = 3;
  constexpr size_t weight_num_two = 2;
  const std::vector<TypeId> valid_dtypes = {TypeId::kNumberTypeBFloat16, TypeId::kNumberTypeFloat16};

  int64_t k_len = -1;
  for (const auto &node : node_list) {
    AnfNodePtrList user_matmuls;
    if (node_users_map.find(node) == node_users_map.end()) continue;
    bool can_process = true;
    for (const auto &user_pair : node_users_map.at(node)) {
      if (IsPrimitiveCNode(user_pair.first, prim::kPrimMatMul) && user_pair.second == 1) {
        auto curr_matmul = user_pair.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(curr_matmul);
        auto input_size = curr_matmul->inputs().size();
        can_process &= CheckFusionValid(curr_matmul, &k_len, input_size - 2, input_size - 1, valid_dtypes);
        user_matmuls.push_back(user_pair.first);
      }
    }
    if ((user_matmuls.size() != weight_num_three && user_matmuls.size() != weight_num_two) || !can_process) {
      continue;
    }
    SortWeightNodeList(user_matmuls);
    AnfNodePtrList getitems;
    Process("MatMul", node, user_matmuls, &getitems);
    if (!getitems.empty()) {
      for (size_t i = 0; i < getitems.size(); i++) {
        (void)mng->Replace(user_matmuls[i], getitems[i]);
      }
      changed = true;
    }
  }
  return changed;
}

void InferenceMultiMatmulWithSplitFusion::Process(const std::string &name, const AnfNodePtr &node,
                                                  const AnfNodePtrList &users, AnfNodePtrList *getitems) const {
  auto kernel_graph = node->func_graph()->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  AnfNodePtrList fused_inputs = {NewValueNode(std::make_shared<Primitive>(name)), node};
  abstract::AbstractBasePtrList new_abs;

  std::vector<void *> data_c_list;
  int64_t k_len = 0;
  int64_t n_len = 0;
  int64_t m_len = 0;
  std::vector<int64_t> n_len_list;
  bool need_rank_offset = false;
  TypeId w_type_id = kTypeUnknown;

  for (auto &user : users) {
    auto matmul = user->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(matmul);
    auto weight_node = matmul->inputs()[2];

    auto w_param = GetParamFromLoad(weight_node->cast<CNodePtr>(), true);
    MS_EXCEPTION_IF_NULL(w_param);
    data_c_list.push_back(reinterpret_cast<void *>(w_param->data_c()));
    auto origin_shape = w_param->shape();
    auto shape = common::AnfAlgo::GetOutputInferShape(weight_node, kIndex0);
    if (origin_shape[0] != shape[0]) {
      need_rank_offset = true;
    }
    k_len = shape[1];
    n_len += shape[0];
    n_len_list.push_back(shape[0]);
    w_type_id = static_cast<TypeId>(w_param->data_type_c());
    new_abs.push_back(user->abstract());
  }

  // Create a new weight tensor concat all of weights
  TypePtr w_dtype = (w_type_id == TypeId::kNumberTypeBFloat16) ? kBFloat16 : kFloat16;
  std::vector<int64_t> new_weight_shape = {n_len, k_len};
  std::shared_ptr<ValueNode> weight_value_node;
  auto anf_node = common::AnfAlgo::GetInputNode(users[0]->cast<CNodePtr>()->inputs()[2]->cast<CNodePtr>(), kIndex0);
  auto para = anf_node->cast<ParameterPtr>();
  if (weights_with_split_cache_map.find(para->name()) != weights_with_split_cache_map.end()) {
    weight_value_node = weights_with_split_cache_map[para->name()];
  } else {
    auto global_rank_id = distributed::collective::CollectiveManager::instance()->global_rank_id();
    weight_value_node = CreateWeightTensor(w_type_id, new_weight_shape, data_c_list, n_len_list, k_len, w_dtype,
                                           need_rank_offset, global_rank_id);
    weights_with_split_cache_map[para->name()] = weight_value_node;
  }

  fused_inputs.push_back(weight_value_node);
  kernel_graph->AddValueNodeToGraph(weight_value_node);
  auto first_matmul = users[0]->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(first_matmul);
  fused_inputs.insert(fused_inputs.end(), first_matmul->inputs().begin() + kIndex3, first_matmul->inputs().end());
  auto fused_matmul = node->func_graph()->NewCNode(fused_inputs);

  m_len = common::AnfAlgo::GetOutputInferShape(users[kIndex0], kIndex0)[kIndex0];
  ShapeVector out_shape = {m_len, n_len};
  auto abs_shape_ptr = std::make_shared<abstract::Shape>(abstract::Shape(out_shape));
  fused_matmul->set_abstract(std::make_shared<abstract::AbstractTensor>(TypeIdToType(w_type_id), abs_shape_ptr));
  AnfNodePtrList split_inputs = {NewValueNode(std::make_shared<Primitive>("SplitWithSize")), fused_matmul};
  auto new_input = opt::CreateValueNodeWithKernelInfo(node->func_graph(), MakeValue<std::vector<int64_t>>(n_len_list));
  auto new_input2 = opt::CreateValueNodeWithKernelInfo(node->func_graph(), MakeValue<int64_t>(1));
  split_inputs.push_back(new_input);
  split_inputs.push_back(new_input2);
  auto split_node = node->func_graph()->NewCNode(split_inputs);
  split_node->set_abstract(std::make_shared<abstract::AbstractTuple>(new_abs));
  for (size_t i = 0; i < users.size(); i++) {
    // create getitem(i)
    auto idx_val = MakeValue(SizeToLong(i));
    auto idx = NewValueNode(idx_val);
    idx->set_abstract(idx_val->ToAbstract());
    auto getitem = node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), split_node, idx});
    (void)getitems->emplace_back(getitem);
  }
}
}  // namespace opt
}  // namespace mindspore
