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

#include "frontend/parallel/parameter_manager.h"

#include <inttypes.h>
#include <sys/time.h>
#include <algorithm>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "base/core_ops.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/node_check.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "utils/trace_base.h"
#include "utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "mindspore/core/utils/parallel_node_check.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
static ParameterUsersInfo FindRefKeyNodeUsers(const RefKeyPair &ref_key_pair, bool (*IsCareNode)(const CNodePtr &)) {
  // Dealing with the RefKey case
  ParameterUsersInfo parameter_user_info;
  auto refkeys = ref_key_pair.second;
  auto cnode = ref_key_pair.first;

  auto cnode_ptr = cnode->cast<CNodePtr>();
  if ((cnode_ptr == nullptr) || !IsValueNode<Primitive>(cnode_ptr->input(0)) || !IsCareNode(cnode_ptr)) {
    return parameter_user_info;
  }

  if (refkeys.size() > 1) {
    MS_LOG(EXCEPTION) << "CNode: " << cnode->fullname_with_scope() << "'s inputs have more than 1 RefKeys";
  }
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  auto cnode_func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(cnode->func_graph()->manager());

  // Find the RefKey being used
  auto candidate_set_by_refkey = cnode_func_graph->manager()->node_users()[refkeys[0]];
  for (auto &candidate : candidate_set_by_refkey) {
    auto candidate_node = candidate.first;
    auto c = candidate_node->cast<CNodePtr>();
    if ((c == nullptr) || !IsValueNode<Primitive>(c->input(0)) || !IsCareNode(c)) {
      continue;
    }
    parameter_user_info.second.second.insert(candidate);
  }

  // Find the corresponding Parameter being used
  std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(refkeys[0], cnode_func_graph);
  if (parameters.size() != 1) {
    MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
  }
  parameter_user_info.first = parameters[0]->cast<ParameterPtr>()->name();
  parameter_user_info.second.first = parameters[0];
  auto candidate_set_by_para = cnode_func_graph->manager()->node_users()[parameters[0]];
  for (auto &candidate : candidate_set_by_para) {
    auto candidate_node = candidate.first;
    auto c = candidate_node->cast<CNodePtr>();
    if ((c == nullptr) || !IsValueNode<Primitive>(c->input(0)) || !IsCareNode(c)) {
      continue;
    }
    parameter_user_info.second.second.insert(candidate);
  }
  return parameter_user_info;
}

static ParameterUsersInfo FindParameterNodeUsers(const AnfNodePtr &node) {
  // In this case, node is a Parameter
  ParameterUsersInfo parameter_user_info;
  MS_EXCEPTION_IF_NULL(node->func_graph());
  MS_EXCEPTION_IF_NULL(node->func_graph()->manager());
  auto candidate_set = node->func_graph()->manager()->node_users()[node];
  for (auto &candidate : candidate_set) {
    auto candidate_node = candidate.first;
    if (IsPrimitiveCNode(candidate_node, prim::kPrimLoad)) {
      if (candidate.second != 1) {
        continue;
      }
      auto load_node_users = node->func_graph()->manager()->node_users()[candidate_node];
      for (auto &node_user : load_node_users) {
        auto cnode = node_user.first->cast<CNodePtr>();
        if (cnode == nullptr || !cnode->has_user_data<OperatorInfo>() || IsSomePrimitive(cnode, RECEIVE)) {
          continue;
        }
        parameter_user_info.second.second.insert(node_user);
      }
    } else {
      auto c = candidate_node->cast<CNodePtr>();
      if (c == nullptr || !c->has_user_data<OperatorInfo>() || IsSomePrimitive(c, RECEIVE)) {
        continue;
      }
      parameter_user_info.second.second.insert(candidate);
    }
  }
  parameter_user_info.first = node->cast<ParameterPtr>()->name();
  parameter_user_info.second.first = node;
  return parameter_user_info;
}

static RefKeyPair CNodeWithRefKeys(const AnfNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> refkeys;
  if (cnode->isa<CNode>()) {
    auto cnode_ptr = cnode->cast<CNodePtr>();
    auto inputs = cnode_ptr->inputs();
    for (auto &one_input : inputs) {
      if (IsValueNode<RefKey>(one_input)) {
        refkeys.push_back(one_input);
      }
    }
    if (refkeys.size() >= 1) {
      return std::make_pair(cnode, refkeys);
    }
  }
  return {nullptr, refkeys};
}

ParameterUsersInfo FindParameterUsers(const AnfNodePtr &node, bool (*IsCareNode)(const CNodePtr &)) {
  ParameterUsersInfo parameter_users_info;

  auto cnode_with_refkeys = CNodeWithRefKeys(node);
  if (cnode_with_refkeys.first != nullptr) {
    // the node is a ref key node
    return FindRefKeyNodeUsers(cnode_with_refkeys, IsCareNode);
  } else if (node->isa<Parameter>()) {
    // the node is a parameter node
    return FindParameterNodeUsers(node);
  }

  return parameter_users_info;
}

static bool IsUsedParameter(const FuncGraphPtr &graph, const AnfNodePtr &parameter, size_t max_depth) {
  if (max_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(EXCEPTION) << "Recursive call is larger than 100000.";
  }
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parameter);
  auto manager = graph->manager();
  auto node_users = manager->node_users()[parameter];
  if (node_users.empty()) {
    return false;
  }
  for (auto node_user : node_users) {
    auto use_node = node_user.first->cast<CNodePtr>();
    if (IsValueNode<FuncGraph>(use_node->input(0))) {
      auto graph_sub = GetValueNode<FuncGraphPtr>(use_node->input(0));
      auto parameters = graph_sub->parameters();
      auto parameter_sub = parameters[IntToSize(node_user.second - 1)];
      return IsUsedParameter(graph_sub, parameter_sub, max_depth + 1);
    }
    if (use_node->input(0)->isa<CNode>()) {
      auto cnode = use_node->input(0)->cast<CNodePtr>();
      if (!IsSomePrimitive(cnode, J) || !IsValueNode<FuncGraph>(cnode->input(1))) {
        return true;
      }
      auto graph_sub = GetValueNode<FuncGraphPtr>(cnode->input(1));
      auto parameters = graph_sub->parameters();
      auto parameter_sub = parameters[IntToSize(node_user.second - 1)];
      return IsUsedParameter(graph_sub, parameter_sub, max_depth + 1);
    }
    return true;
  }
  return true;
}

static RankList GetGroupByTensorInfo(const TensorInfo &tensor_info) {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  RankList stage_device_list = g_device_manager->GetDeviceListInThisStage();
  Shape dev_matrix_shape = tensor_info.tensor_layout().device_arrangement().array();
  Shape tensor_map = tensor_info.tensor_layout().tensor_map().array();

  DeviceMatrix dev_matrix(rank, stage_device_list, dev_matrix_shape);
  RankList group_devices;
  if (dev_matrix.GetDevicesByTensorMap(tensor_map, &group_devices) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Get devices by tensor map failed";
  }

  std::sort(group_devices.begin(), group_devices.end());
  return group_devices;
}

static ParameterSliceInfo GetParameterSliceInfo(const std::pair<AnfNodePtr, int64_t> &param_info) {
  auto user_cnode = param_info.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(user_cnode);
  auto user_input_index = param_info.second;
  OperatorInfoPtr op_info = user_cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(op_info);

  TensorInfo tensor_info;
  if (IsPrimitiveCNode(user_cnode, prim::kPrimSend)) {
    auto param_index = IntToSize(GetValue<int>(user_cnode->GetPrimalAttr(PARAM_INDEX)));
    tensor_info = op_info->inputs_tensor_info()[param_index];
  } else {
    size_t input_tensor_info_size = op_info->inputs_tensor_info().size();
    if (SizeToLong(input_tensor_info_size) <= user_input_index - 1) {
      MS_LOG(EXCEPTION) << op_info->name() << ": the size of inputs tensor info is " << input_tensor_info_size
                        << ", but the index is " << (user_input_index - 1);
    }
    tensor_info = op_info->inputs_tensor_info()[LongToSize(user_input_index - 1)];
  }

  ParameterSliceInfo parameter_slice_info;
  parameter_slice_info.slice_shape = tensor_info.slice_shape();
  parameter_slice_info.group_ranks = GetGroupByTensorInfo(tensor_info);
  MS_LOG(DEBUG) << "The op name is " << op_info->name() << ", the parameter index is " << (user_input_index - 1)
                << ", the slice shape is " << tensor_info.slice_shape() << ", the origin shape is "
                << tensor_info.shape() << ", the group rank list is " << parameter_slice_info.group_ranks;
  return parameter_slice_info;
}

void CheckParameterSplit(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    ParameterUsersInfo parameter_users_info = FindParameterUsers(node, IsParallelCareNode);
    auto &users_set = parameter_users_info.second.second;
    if (users_set.size() <= 1) {
      continue;
    }

    auto parameter_name = parameter_users_info.first;
    MS_LOG(INFO) << "The parameter: " << parameter_name << " has " << users_set.size() << " users";
    auto &first_user = users_set.front();
    ParameterSliceInfo parameter_slice_info = GetParameterSliceInfo(first_user);
    Shape first_user_slice_shape = parameter_slice_info.slice_shape;
    RankList first_user_group_list = parameter_slice_info.group_ranks;

    for (auto iter = users_set.begin() + 1; iter != users_set.end(); ++iter) {
      auto &user = *iter;
      ParameterSliceInfo user_slice_info = GetParameterSliceInfo(user);
      Shape user_slice_shape = user_slice_info.slice_shape;
      RankList user_group_list = user_slice_info.group_ranks;
      if (first_user_slice_shape != user_slice_shape) {
        MS_LOG(EXCEPTION) << "The parameter: " << parameter_name
                          << " has multiple users, but the slice shapes are different";
      }

      if (ParallelContext::GetInstance()->pipeline_stage_split_num() == 1 && first_user_group_list != user_group_list) {
        MS_LOG(EXCEPTION) << "The parameter: " << parameter_name
                          << " has multiple users, but the group rank list are different, "
                          << "the group rank list for first user is " << first_user_group_list
                          << ", and the group rank list for this user is " << user_group_list;
      }
    }
  }
}

namespace {
void RevertSymbolicKeyInstance(const FuncGraphPtr &root, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(node);
  auto symbolic_key = GetValueNode<SymbolicKeyInstancePtr>(node);
  MS_EXCEPTION_IF_NULL(symbolic_key);
  auto all_upstream_node = root->manager()->node_users()[node];
  for (auto &upstream_node : all_upstream_node) {
    FuncGraphPtr fg = upstream_node.first->func_graph();
    if (symbolic_key->node()->isa<Parameter>()) {
      for (auto &param : root->parameters()) {
        if (*param == *symbolic_key->node()) {
          AnfNodePtr reverted_node = root->NewCNode({NewValueNode(prim::kPrimEmbed), param});
          MS_EXCEPTION_IF_NULL(reverted_node);
          MS_LOG(DEBUG) << "before replace " << node->ToString() << " to node " << reverted_node->DebugString();
          (void)fg->manager()->Replace(node, reverted_node);
          MS_LOG(DEBUG) << "revert node " << node->ToString() << " to node " << reverted_node->DebugString();
        }
      }
    }
  }
}
}  // namespace

void HandleSymbolicKeyInstance(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  MS_EXCEPTION_IF_NULL(root);
  for (auto &node : all_nodes) {
    // revert back SymbolicKeyInstance to embed() primitive
    if (IsValueNode<SymbolicKeyInstance>(node)) {
      RevertSymbolicKeyInstance(root, node);
      continue;
    }
  }
}

bool ParameterIsCloned(const AnfNodePtr &parameter_node) {
  MS_EXCEPTION_IF_NULL(parameter_node);
  auto cloned_parameter = parameter_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(cloned_parameter);

  // find the clone parameter
  if (!cloned_parameter->has_default()) {
    return false;
  }
  auto param_value = cloned_parameter->param_info();
  if (param_value == nullptr) {
    return false;
  }
  bool cloned = param_value->cloned();
  if (!cloned) {
    return false;
  }

  MS_LOG(INFO) << "The parameter: " << cloned_parameter->name() << " is cloned";
  return true;
}

void HandleNoUsedParameter(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  bool full_batch = ParallelContext::GetInstance()->full_batch();
  if (full_batch) {
    return;
  }

  // in grad accumulation mode, if use dynamic lr, it has some parameters in optimizer which no used for first graph,
  // but used for second graph(such as global_step), so can not change their shapes
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  if (grad_accumulation_step > 1) {
    MS_LOG(INFO) << "In grad accumulation mode, do not handle no used parameters";
    return;
  }

  auto dev_num = g_device_manager->stage_device_num();
  auto parameters = root->parameters();
  for (auto &parameter : parameters) {
    if (IsUsedParameter(root, parameter, 0)) {
      continue;
    }
    auto parameter_shape = GetNodeShape(parameter);
    if (parameter_shape.empty()) {
      continue;
    }
    Shape slice_shape = parameter_shape[0];
    if (slice_shape.empty()) {
      continue;
    }
    slice_shape[0] = slice_shape[0] / dev_num;
    auto slice_shape_ptr = std::make_shared<abstract::Shape>(slice_shape);
    auto abstract = parameter->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    auto abstract_cloned = abstract->Clone();
    MS_EXCEPTION_IF_NULL(abstract_cloned);
    abstract_cloned->set_shape(slice_shape_ptr);
    parameter->set_abstract(abstract_cloned);
  }
}

static bool IsFullySplitParameter(const ParameterPtr &param_ptr) {
  auto tensor_layout = param_ptr->user_data<parallel::TensorLayout>();
  if (tensor_layout == nullptr) {
    return false;
  }

  auto dev_mat_shape = tensor_layout->device_arrangement().array();
  auto tensor_map = tensor_layout->tensor_map().array();
  int64_t rank = g_device_manager->global_rank();
  RankList rank_list = g_device_manager->GetDeviceListInThisStage();
  DeviceMatrix dev_matrix(rank, rank_list, dev_mat_shape);
  RankList group_devices;
  if (dev_matrix.GetDevicesByTensorMap(tensor_map, &group_devices) != SUCCESS) {
    MS_LOG(WARNING) << "Get devices by tensor map failed, invalid tensor layout";
    return false;
  }

  if (group_devices.size() == 1) {
    MS_LOG(INFO) << "The parameter: " << param_ptr->name() << " is fully split";
    return true;
  }
  return false;
}

static void InsertFullySplitParamGradAccu(const std::pair<AnfNodePtr, int> &node_user,
                                          const FuncGraphManagerPtr &manager, const AnfNodePtr &accu_parameter) {
  auto cnode = node_user.first->cast<CNodePtr>();
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(WARNING) << cnode->DebugString() << " can not insert fully split param grad accumulation node";
    return;
  }
  OperatorAttrs attrs;
  auto py_instance = CreatOpInstance(attrs, "_VirtualAdd", "grad_accu");
  auto value_node = NewValueNode(py_instance);
  std::vector<AnfNodePtr> virtual_node_input = {value_node, cnode->input(IntToSize(node_user.second)), accu_parameter};
  auto graph = cnode->func_graph();
  auto virtual_node = graph->NewCNode(virtual_node_input);
  manager->SetEdge(cnode, node_user.second, virtual_node);
}

void HandleFullySplitParameters(const FuncGraphPtr &root) {
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  if ((grad_accumulation_step <= 1) || root->has_flag(ACCUMULATION)) {
    return;
  }

  auto parameters = root->parameters();
  auto node_users_map = root->manager()->node_users();
  for (auto &parameter : parameters) {
    auto param_ptr = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);

    if (!IsFullySplitParameter(param_ptr)) {
      continue;
    }

    auto accu_parameter = FindGradAccuParameter(parameters, param_ptr->name());
    if (!accu_parameter) {
      continue;  // some parameters no need to handle, such as itself or lr
    }

    auto node_users = node_users_map[parameter];
    for (auto &user : node_users) {
      auto node = user.first;
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (!cnode->in_forward_flag()) {
        continue;
      }
      InsertFullySplitParamGradAccu(user, root->manager(), accu_parameter);
      MS_LOG(INFO) << "Insert full split assign add node for " << param_ptr->name();
      break;  // only need to insert once, if the parameter has many users
    }
  }
}

void SetClonedTensorShapeForOptimizer(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  for (auto &cloned_parameter_node : root->parameters()) {
    MS_EXCEPTION_IF_NULL(cloned_parameter_node);
    auto cloned_parameter = cloned_parameter_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(cloned_parameter);

    if (!ParameterIsCloned(cloned_parameter_node)) {
      continue;
    }
    auto param_value = cloned_parameter->param_info();
    if (param_value == nullptr) {
      continue;
    }
    // get the cloned index
    int64_t cloned_index = param_value->cloned_index();

    // find the be cloned parameter
    bool found_be_cloned_parameter = false;
    ParameterPtr cloned_from_parameter = nullptr;
    AnfNodePtr cloned_from_node = nullptr;
    for (auto &be_cloned_parameter_node : root->parameters()) {
      MS_EXCEPTION_IF_NULL(be_cloned_parameter_node);
      auto be_cloned_parameter = be_cloned_parameter_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(be_cloned_parameter);
      if (!be_cloned_parameter->has_default()) {
        continue;
      }

      auto param_value_in = be_cloned_parameter->param_info();
      if (param_value_in == nullptr) {
        continue;
      }
      if (!param_value_in->be_cloned()) {
        continue;
      }

      // get the be cloned index
      auto &be_cloned_index = param_value_in->be_cloned_index();
      if (std::find(be_cloned_index.begin(), be_cloned_index.end(), cloned_index) != be_cloned_index.end()) {
        found_be_cloned_parameter = true;
        cloned_from_parameter = be_cloned_parameter;
        cloned_from_node = be_cloned_parameter_node;
      }
    }

    if (found_be_cloned_parameter) {
      // set the shape and tensor layout for cloned parameter
      std::string param_name = cloned_parameter_node->cast<ParameterPtr>()->name();
      if (cloned_from_parameter->user_data<TensorLayout>() == nullptr) {
        MS_LOG(WARNING) << "The parameter " << param_name << " has not tensor layout, skip it";
        continue;
      }
      auto tensor_layout = cloned_from_parameter->user_data<TensorLayout>();
      MS_EXCEPTION_IF_NULL(cloned_parameter_node->abstract());
      MS_EXCEPTION_IF_NULL(cloned_from_node->abstract());
      auto cloned_abstract = cloned_parameter_node->abstract()->Clone();
      MS_EXCEPTION_IF_NULL(cloned_abstract);
      // from pipeline or grad accumulation
      if (param_name.find(ACCU_GRADS) != std::string::npos) {
        auto slice_shape = cloned_from_parameter->user_data<TensorLayout>()->slice_shape().array();
        std::shared_ptr<abstract::BaseShape> parallel_shape = std::make_shared<abstract::Shape>(slice_shape);
        MS_EXCEPTION_IF_NULL(parallel_shape);
        cloned_abstract->set_shape(parallel_shape);
        // in opt shard, accu_grad's shape is different from the original param's shape
        if (ParallelContext::GetInstance()->enable_parallel_optimizer()) {
          TensorLayout new_layout = *tensor_layout;
          new_layout.set_opt_shard_group("");
          tensor_layout = std::make_shared<TensorLayout>(new_layout);
        }
      } else {
        cloned_abstract->set_shape(cloned_from_node->abstract()->GetShapeTrack());
      }
      cloned_parameter->set_user_data<TensorLayout>(tensor_layout);
      cloned_parameter_node->set_abstract(cloned_abstract);
      MS_LOG(INFO) << "The parameter: " << cloned_parameter->name()
                   << " is cloned, the be cloned parameter is: " << cloned_from_parameter->name()
                   << ", clone index is:  " << cloned_index;
    } else {
      MS_LOG(EXCEPTION) << "The parameter: " << cloned_parameter->name() << " is cloned, cloned index is  "
                        << cloned_index << ", but not found the be cloned parameter";
    }
  }
}

void HandleAdaFactorOpt(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  for (auto &param_node : root->parameters()) {
    MS_EXCEPTION_IF_NULL(param_node);
    auto param = param_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    std::string param_name = param->name();
    if (param_name.find(EXP_AVG) != std::string::npos) {
      continue;
    }

    auto tensor_layout = param->user_data<TensorLayout>();
    if (tensor_layout == nullptr) {
      continue;
    }

    int64_t row_col_count = 0;
    int64_t exp_avg_sq_count = 0;
    for (auto &row_col_node : root->parameters()) {
      MS_EXCEPTION_IF_NULL(row_col_node);
      auto row_col_param = row_col_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(row_col_param);
      std::string row_col_param_name = row_col_param->name();
      std::string exp_row_name = EXP_AVG_SQ_ROW + param_name;
      std::string exp_col_name = EXP_AVG_SQ_COL + param_name;
      std::string exp_avg_name = EXP_AVG_SQ + param_name;

      if ((row_col_param_name != exp_row_name) && (row_col_param_name != exp_col_name) &&
          (row_col_param_name != exp_avg_name)) {
        continue;
      }

      auto slice_shape = tensor_layout->slice_shape().array();
      auto shape_size = slice_shape.size();
      bool is_row_or_col_param = (row_col_param_name == exp_row_name) || (row_col_param_name == exp_col_name);
      if (is_row_or_col_param && shape_size <= 1) {
        continue;
      }

      if (row_col_param_name == exp_avg_name && shape_size != 1) {
        continue;
      }

      auto origin_shape = tensor_layout->tensor_shape().array();
      auto dev_mat = tensor_layout->device_arrangement().array();
      auto tensor_map = tensor_layout->tensor_map().array();

      if (row_col_param_name == exp_row_name) {
        slice_shape.pop_back();
        origin_shape.pop_back();
        tensor_map.pop_back();
        row_col_count++;
      } else if (row_col_param_name == exp_col_name) {
        (void)slice_shape.erase(slice_shape.begin() + static_cast<different_type>(SECOND_FROM_END(shape_size)));
        (void)origin_shape.erase(origin_shape.begin() + static_cast<different_type>(SECOND_FROM_END(shape_size)));
        (void)tensor_map.erase(tensor_map.begin() + static_cast<different_type>(SECOND_FROM_END(shape_size)));
        row_col_count++;
      } else {
        exp_avg_sq_count++;
      }

      TensorLayout new_tensor_layout;
      if (new_tensor_layout.InitFromVector(dev_mat, tensor_map, origin_shape) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Init tensor layout failed";
      }

      auto cloned_abstract = row_col_node->abstract()->Clone();
      MS_EXCEPTION_IF_NULL(cloned_abstract);
      std::shared_ptr<abstract::BaseShape> parallel_shape = std::make_shared<abstract::Shape>(slice_shape);
      MS_EXCEPTION_IF_NULL(parallel_shape);
      cloned_abstract->set_shape(parallel_shape);
      row_col_param->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(new_tensor_layout));
      row_col_node->set_abstract(cloned_abstract);
      MS_LOG(INFO) << "Set the slice shape for " << row_col_param_name << ", origin shape is " << origin_shape
                   << ", new slice shape is " << slice_shape;

      if (row_col_count == 2 || exp_avg_sq_count == 1) {
        break;
      }
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
