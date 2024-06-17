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
#include "plugin/device/ascend/optimizer/ir_fusion/inference_multi_matmul_fusion.h"

#include <vector>
#include <map>
#include <algorithm>
#include <functional>
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
static std::map<std::string, ValueNodePtr> weights_cache_map;
static bool global_mmf_flag = true;

bool InferenceMultiMatmulFusion::Run(const FuncGraphPtr &graph) {
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  bool changed = false;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return false;
  }

  constexpr auto kInferenceMultiMatmulOpName = "InferenceMultiMatmul";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_opt =
    (std::find(enable_op_list.begin(), enable_op_list.end(), kInferenceMultiMatmulOpName) != enable_op_list.end());
  if (!enable_opt) {
    return false;
  }

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
      // the node is MatMul's first input.
      if (IsPrimitiveCNode(user_pair.first, prim::kPrimMatMul) && user_pair.second == 1) {
        auto curr_matmul = user_pair.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(curr_matmul);
        auto input_size = curr_matmul->inputs().size();
        can_process &= CheckFusionValid(curr_matmul, &k_len, input_size - 2, input_size - 1, valid_dtypes);
        user_matmuls.push_back(user_pair.first);
      }
    }
    global_mmf_flag &= can_process;
    if (user_matmuls.size() <= 1 || !global_mmf_flag) {
      continue;
    }
    SortWeightNodeList(&user_matmuls);
    AnfNodePtrList getitems;
    if (user_matmuls.size() == weight_num_two) {
      Process("MatmulSplitOut2", node, user_matmuls, &getitems);
    } else if (user_matmuls.size() == weight_num_three) {
      Process("MatmulSplitOut3", node, user_matmuls, &getitems);
    }
    if (!getitems.empty()) {
      for (size_t i = 0; i < getitems.size(); i++) {
        (void)mng->Replace(user_matmuls[i], getitems[i]);
      }
      changed = true;
    }
  }
  return changed;
}

void InferenceMultiMatmulFusion::Process(const std::string &name, const AnfNodePtr &node, const AnfNodePtrList &users,
                                         AnfNodePtrList *getitems) const {
  auto kernel_graph = node->func_graph()->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  AnfNodePtrList fused_inputs = {NewValueNode(std::make_shared<Primitive>(name)), node};
  abstract::AbstractBasePtrList new_abs;

  TypeId w_type_id = kTypeUnknown;
  std::vector<void *> data_c_list;
  int64_t k_len = 0;
  int64_t n_len = 0;
  std::vector<int64_t> n_len_list;
  bool need_rank_offset = false;

  for (auto &user : users) {
    auto matmul = user->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(matmul);
    auto weight_node = matmul->inputs()[2]->cast<CNodePtr>();

    auto w_param = GetParamFromLoad(weight_node, true);
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
  if (weights_cache_map.find(para->name()) != weights_cache_map.end()) {
    weight_value_node = weights_cache_map[para->name()];
  } else {
    auto global_rank_id = distributed::collective::CollectiveManager::instance()->global_rank_id();
    weight_value_node = CreateWeightTensor(w_type_id, new_weight_shape, data_c_list, n_len_list, k_len, w_dtype,
                                           need_rank_offset, global_rank_id);
    weights_cache_map[para->name()] = weight_value_node;
  }

  fused_inputs.push_back(weight_value_node);
  kernel_graph->AddValueNodeToGraph(weight_value_node);

  auto fused_matmul = node->func_graph()->NewCNode(fused_inputs);
  common::AnfAlgo::SetNodeAttr(n_lens_str, MakeValue(n_len_list), fused_matmul);
  const bool is_fixed_weight = true;
  common::AnfAlgo::SetNodeAttr(is_fixed_weight_str, MakeValue<bool>(is_fixed_weight), fused_matmul);

  fused_matmul->set_abstract(std::make_shared<abstract::AbstractTuple>(new_abs));
  for (size_t i = 0; i < users.size(); i++) {
    auto idx_val = MakeValue(SizeToLong(i));
    auto idx = NewValueNode(idx_val);
    idx->set_abstract(idx_val->ToAbstract());
    auto getitem = node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fused_matmul, idx});
    (void)getitems->emplace_back(getitem);
  }
}
}  // namespace opt
}  // namespace mindspore
