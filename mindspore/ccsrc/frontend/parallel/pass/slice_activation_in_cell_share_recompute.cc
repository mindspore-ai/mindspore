/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/slice_activation_in_cell_share_recompute.h"
#include <memory>
#include <string>
#include <vector>
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/tensor_layout/construct_operator.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace parallel {
namespace {
CNodePtr CreateStridedSliceCNode(const parallel::Shape &begin, const parallel::Shape &end,
                                 const parallel::Shape &strides, const AnfNodePtr &node) {
  auto slice_op = parallel::CreateStridedSliceOp(0, begin, end, strides);
  auto slice_input = parallel::CreateInput(slice_op, node, parallel::STRIDEDSLICE);
  auto func_graph = node->func_graph();
  CNodePtr new_node = func_graph->NewCNode(slice_input);
  return new_node;
}

CNodePtr CreateAllGatherCNode(const AnfNodePtr &node, const std::string &group) {
  auto op = parallel::CreateAllGatherOp(group);
  auto allgather_input = parallel::CreateInput(op, node, "recompute_slice_allgather");
  auto func_graph = node->func_graph();
  CNodePtr new_node = func_graph->NewCNode(allgather_input);
  return new_node;
}

std::vector<parallel::Group> InferRepeatedRankList(const CNodePtr &cnode) {
  OperatorInfoPtr operator_info = cnode->user_data<parallel::OperatorInfo>();
  std::vector<parallel::TensorInfo> output_info = operator_info->outputs_tensor_info();
  if (output_info.size() != 1) {
    MS_LOG(WARNING) << "The output_info size is wrong, node is" << cnode->DebugString();
    return std::vector<parallel::Group>();
  }
  auto tensor_layout = output_info[0].tensor_layout();
  auto tensor_map = tensor_layout.origin_tensor_map();
  std::vector<parallel::Group> groups;
  (void)operator_info->CreateGroupByTensorMap(tensor_map.array(), &groups);
  return groups;
}

CNodePtr CreateDependNode(const AnfNodePtr &src_node, const AnfNodePtr &rely_node, const std::string &attr_name = "") {
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), src_node, rely_node};
  auto depend_node = src_node->func_graph()->NewCNode(depend_inputs);
  depend_node->set_abstract(src_node->abstract()->Clone());
  if (!attr_name.empty()) {
    depend_node->AddAttr(attr_name, MakeValue(true));
  }
  return depend_node;
}

CNodePtr CreateAllGatherNode(const AnfNodePtr &activation_cnode, const std::vector<parallel::Group> &groups) {
  if (groups.empty()) {
    return nullptr;
  }
  auto group = groups[0];
  if (group.GetDevNum() == 0) {
    MS_LOG(ERROR) << "The dev num of group should not be 0.";
    return nullptr;
  }
  CNodePtr allgather_cnode = CreateAllGatherCNode(activation_cnode, group.name());
  allgather_cnode->set_abstract(activation_cnode->abstract()->Clone());
  auto ag_input_shape = allgather_cnode->abstract()->BuildShape();
  auto input_shape_element = ag_input_shape->cast<abstract::ShapePtr>()->shape();
  if (!input_shape_element.empty()) {
    input_shape_element[0] = input_shape_element[0] * group.GetDevNum();
  }
  auto ag_output_shape = std::make_shared<abstract::Shape>(input_shape_element);
  allgather_cnode->abstract()->set_shape(ag_output_shape);
  allgather_cnode->AddAttr("recompute_allgather", MakeValue(true));
  return allgather_cnode;
}

CNodePtr CreateSliceNode(const CNodePtr &activation_cnode, const std::vector<parallel::Group> &groups) {
  auto output_shape = activation_cnode->abstract()->BuildShape();
  std::vector<int64_t> out_shape_element = output_shape->cast<abstract::ShapePtr>()->shape();
  if (out_shape_element.empty()) {
    return nullptr;
  }
  int64_t global_rank_id = parallel::g_device_manager->global_rank();
  int64_t stage_num = parallel::g_device_manager->stage_num();
  int64_t device_num = SizeToLong(parallel::g_device_manager->DeviceNum());
  int64_t stage_device_num = device_num / stage_num;
  int64_t local_rank_id = global_rank_id % stage_device_num;
  auto group = groups[0];
  if (group.GetDevNum() == 0) {
    MS_LOG(ERROR) << "The dev num of group should not be 0.";
    return nullptr;
  }
  if (out_shape_element[0] % SizeToLong(group.GetDevNum()) != 0) {
    MS_LOG(WARNING) << "The output_shape first dim:" << out_shape_element[0]
                    << " cannot be divisible by the repeated size: " << group.GetDevNum()
                    << "The slice would not activate to this node: " << activation_cnode->DebugString();
    return nullptr;
  }
  int64_t group_deivce_num = SizeToLong(group.GetDevNum());
  if (group_deivce_num == 0) {
    MS_LOG(ERROR) << "The device num of group should not be 0.";
    return nullptr;
  }
  std::vector<int64_t> slice_begin(out_shape_element.size(), 0);
  slice_begin[0] = (local_rank_id % group_deivce_num) * (out_shape_element[0] / group_deivce_num);
  std::vector<int64_t> slice_end = out_shape_element;
  slice_end[0] = (local_rank_id % group_deivce_num + 1) * (out_shape_element[0] / group_deivce_num);
  std::vector<int64_t> slice_strides(out_shape_element.size(), 1);
  CNodePtr slice_cnode = CreateStridedSliceCNode(slice_begin, slice_end, slice_strides, activation_cnode);
  slice_cnode->set_abstract(activation_cnode->abstract()->Clone());
  std::vector<int64_t> slice_shape = out_shape_element;
  slice_shape[0] = out_shape_element[0] / group_deivce_num;
  std::shared_ptr<abstract::BaseShape> slice_base_shape = std::make_shared<abstract::Shape>(slice_shape);
  slice_cnode->abstract()->set_shape(slice_base_shape);
  slice_cnode->AddAttr("recompute_slice", MakeValue(true));
  return slice_cnode;
}

void PrintAnfNodeInfo(const AnfNodePtr &anf_node, const std::string &node_info) {
  std::string unique_id;
  if (anf_node->isa<CNode>() && anf_node->cast<CNodePtr>()->HasPrimalAttr("unique_id")) {
    unique_id = GetValue<std::string>(anf_node->cast<CNodePtr>()->GetPrimalAttr("unique_id"));
  }
  MS_LOG(INFO) << "The node " << node_info << " debug string is:" << anf_node->DebugString()
               << ", fullname is:" << anf_node->fullname_with_scope() << ", unique id is:" << unique_id;
}

bool is_step_in() {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return false;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  return cell_reuse;
}

CNodePtr SliceRelyNode(const std::shared_ptr<FuncGraphManager> &manager, const CNodePtr &node,
                       const AnfNodePtr anf_node) {
  CNodePtr slice_rely_node = nullptr;
  auto node_usrers_map = manager->node_users()[node];
  for (const auto &activation_user_pair : node_usrers_map) {
    if (activation_user_pair.first == anf_node) {
      continue;
    }
    if (!activation_user_pair.first->isa<CNode>()) {
      continue;
    }
    if (activation_user_pair.first->func_graph() != node->func_graph()) {
      continue;
    }
    slice_rely_node = activation_user_pair.first->cast<CNodePtr>();
    PrintAnfNodeInfo(slice_rely_node, "slice_rely_node");
    break;
  }
  return slice_rely_node;
}
}  // namespace

void SliceReuseRecomputedActivationNodes(const FuncGraphPtr &graph) {
  if (!is_step_in()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodePtr ret = graph->get_return();
  std::vector<AnfNodePtr> origin_nodes_topological = DeepScopedGraphSearch(ret);
  for (auto &anode : origin_nodes_topological) {
    if (!anode->isa<CNode>()) {
      continue;
    }
    auto node = anode->cast<CNodePtr>();
    if (!IsValueNode<FuncGraph>(node->input(0))) {
      continue;
    }
    auto recompute_graph = GetValueNode<FuncGraphPtr>(node->input(0));
    if (!recompute_graph->has_flag("recompute_k_graph")) {
      continue;
    }
    for (size_t i = 1; i < node->size(); ++i) {
      auto anf_node = node->input(i);
      if (IsPrimitiveCNode(anf_node, prim::kPrimDepend)) {
        auto c_depend_node = anf_node->cast<CNodePtr>();
        anf_node = c_depend_node->input(1);
      }
      if (anf_node->isa<Parameter>()) {
        continue;
      }
      auto real_node = GetRealKernelNode(anf_node, -1).first;
      PrintAnfNodeInfo(real_node, "real_node");
      if (!real_node->isa<CNode>()) {
        continue;
      }
      auto real_cnode = real_node->cast<CNodePtr>();
      if (!real_cnode->HasAttr(kAttrSliceActivation) || IsPrimitiveCNode(node, prim::kPrimTranspose) ||
          !real_cnode->has_user_data<parallel::OperatorInfo>()) {
        MS_LOG(INFO) << "node->HasAttr(kAttrSliceActivation):" << real_cnode->HasAttr(kAttrSliceActivation)
                     << ", node->has_user_data<parallel::OperatorInfo>()"
                     << real_cnode->has_user_data<parallel::OperatorInfo>();
        continue;
      }
      if (!anf_node->isa<CNode>()) {
        continue;
      }
      CNodePtr slice_rely_node = SliceRelyNode(manager, node, anf_node);
      auto groups = InferRepeatedRankList(real_cnode);
      if (groups.empty()) {
        continue;
      }
      auto slice_pos_node = anf_node->cast<CNodePtr>();
      auto slice_cnode = CreateSliceNode(slice_pos_node, groups);
      if (!slice_cnode) {
        MS_LOG(INFO) << "Create slice failed";
        continue;
      }
      manager->SetEdge(node, i, slice_cnode);
      // create depend for slice
      if (!slice_rely_node) {
        auto slice_depend = CreateDependNode(slice_rely_node, slice_cnode, "slice_activation_depend");
        (void)manager->Replace(slice_rely_node, slice_depend);
      }

      // handle allgather node
      auto recompute_graph_inputs = recompute_graph->parameters();
      auto ref_node = recompute_graph_inputs[i - 1];
      ref_node->set_abstract(slice_cnode->abstract()->Clone());
      PrintAnfNodeInfo(ref_node, "ref_node");
      if (i > 1) {
        auto allgather_rely_node = recompute_graph_inputs.front();
        auto allgather_depend = CreateDependNode(ref_node, allgather_rely_node, "allgather_activation_depend");
        (void)manager->Replace(ref_node, allgather_depend);
        ref_node = allgather_depend;
      }
      auto allgather_cnode = CreateAllGatherNode(ref_node, groups);
      (void)manager->Replace(ref_node, allgather_cnode);
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
