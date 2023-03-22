/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <cinttypes>
#include <algorithm>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "utils/hash_map.h"
#include "mindspore/core/ops/core_ops.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/get_parallel_info.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/node_check.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "utils/trace_base.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "pipeline/jit/pipeline.h"
#include "mindspore/core/utils/parallel_node_check.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
using TensorLayoutPtr = std::shared_ptr<TensorLayout>;
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
        if (IsSomePrimitive(cnode, DEPEND)) {
          auto depend_node_users = node->func_graph()->manager()->node_users()[node_user.first];
          for (auto depend_user : depend_node_users) {
            if (IsPrimitiveCNode(depend_user.first, prim::kPrimLoad)) {
              auto local_load_node_users = node->func_graph()->manager()->node_users()[depend_user.first];
              for (auto local_load_user : local_load_node_users) {
                auto local_cnode = local_load_user.first->cast<CNodePtr>();
                if (local_cnode == nullptr || !local_cnode->has_user_data<OperatorInfo>() ||
                    IsSomePrimitive(local_cnode, RECEIVE)) {
                  continue;
                }
                parameter_user_info.second.second.insert(local_load_user);
              }
            }
          }
        }

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
    auto param_ptr = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    // the node is a parameter node
    if (param_ptr->has_default()) {
      return FindParameterNodeUsers(node);
    }
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
    auto parameter_tensor_info = GetInputsTensorInfo(first_user);

    for (auto iter = users_set.begin() + 1; iter != users_set.end(); ++iter) {
      auto &user = *iter;
      auto user_tensor_info = GetInputsTensorInfo(user);
      if (parameter_tensor_info == user_tensor_info) {
        continue;
      } else {
        MS_LOG(EXCEPTION) << "The parameter: " << parameter_name
                          << " has multiple users, but the TensorInfo are different, they are "
                          << parameter_tensor_info.tensor_layout().ToString() << std::endl
                          << " and " << user_tensor_info.tensor_layout().ToString();
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
    if (slice_shape.empty() || slice_shape[0] < dev_num) {
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

bool IsFullySplitParameter(const ParameterPtr &param_ptr, size_t allow_repeat_num) {
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

  if (group_devices.size() <= allow_repeat_num) {
    MS_LOG(INFO) << "The parameter: " << param_ptr->name() << " is fully split";
    return true;
  }
  return false;
}

py::object GetPyParameterObj(const ParamInfoPtr &param_info, const std::string &obj) {
  py::object py_obj = py::cast(param_info);
  if (py::isinstance<py::none>(py_obj)) {
    return py::none();
  }
  return python_adapter::GetPyObjAttr(py_obj, obj);
}

void SliceParameterObj(const ParameterPtr &parameter, const TensorLayoutPtr &tensor_layout) {
  auto param_info = parameter->param_info();
  if (param_info == nullptr) {
    MS_LOG(WARNING) << "parameter: " << parameter->DebugString() << " doesn't have param_info.";
    return;
  }
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  auto phase = graph_executor->phase();
  auto py_obj = GetPyParameterObj(param_info, OBJ);
  if (py::isinstance<py::none>(py_obj)) {
    MS_LOG(WARNING) << "Parameter: " << parameter->DebugString() << " can't find python obj.";
    return;
  }
  if (tensor_layout == nullptr) {
    (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_PARAMETER_FN_NAME, py_obj, py::str(phase),
                                   py::none());
    return;
  }
  // create python layout obj
  const auto &device_arrangement = tensor_layout->device_arrangement().array();
  const auto &tensor_map = tensor_layout->tensor_map().array();
  auto slice_shape = tensor_layout->slice_shape().array();
  int64_t field_size = tensor_layout->get_field_size();
  bool uniform_split = tensor_layout->uniform_split();
  std::string opt_shard_group = tensor_layout->opt_shard_group();
  if (!opt_shard_group.empty()) {
    slice_shape = tensor_layout->opt_shard_slice_shape();
  }
  py::tuple layout =
    py::make_tuple(device_arrangement, tensor_map, slice_shape, field_size, uniform_split, opt_shard_group);

  // Call Python _slice_parameter Fn to slice python parameter obj
  (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_PARAMETER_FN_NAME, py_obj, py::str(phase), layout);

  // handle cloned parameter, like accu_grad and optimizer param
  auto cloned_py_obj = GetPyParameterObj(param_info, CLONED_OBJ);
  if (!py::isinstance<py::none>(cloned_py_obj)) {
    if (!py::isinstance<py::list>(cloned_py_obj)) {
      MS_LOG(EXCEPTION) << "parameter: " << parameter->DebugString() << " doesn't have correct cloned obj";
    }
    auto obj_list = py::cast<py::list>(cloned_py_obj);
    for (size_t i = 0; i < obj_list.size(); ++i) {
      py::object each_cloned_obj = obj_list[i];
      (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_PARAMETER_FN_NAME, each_cloned_obj, py::str(phase),
                                     layout);
    }
  }
}

static void SliceCacheParameterObj(const ParameterPtr &parameter, const py::dict &layout_dict) {
  auto param_info = parameter->param_info();
  if (param_info == nullptr) {
    MS_LOG(WARNING) << "parameter: " << parameter->DebugString() << " doesn't have param_info.";
    return;
  }
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  auto phase = graph_executor->phase();
  auto py_obj = GetPyParameterObj(param_info, OBJ);
  if (py::isinstance<py::none>(py_obj)) {
    MS_LOG(WARNING) << "Parameter: " << parameter->DebugString() << " can't find python obj.";
    return;
  }
  auto name = parameter->name();
  if (!layout_dict.contains(name)) {
    (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, INIT_OPTIMIZER_STATE_FN, py_obj, py::str(phase));
    return;
  }
  auto layout = layout_dict[py::str(name)];
  // Call Python _slice_parameter Fn to slice python parameter obj
  (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_PARAMETER_FN_NAME, py_obj, py::str(phase), layout);

  // handle cloned parameter, like accu_grad and optimizer param
  auto cloned_py_obj = GetPyParameterObj(param_info, CLONED_OBJ);
  if (!py::isinstance<py::none>(cloned_py_obj)) {
    if (!py::isinstance<py::list>(cloned_py_obj)) {
      MS_LOG(EXCEPTION) << "parameter: " << parameter->DebugString() << " doesn't have correct cloned obj";
    }
    auto obj_list = py::cast<py::list>(cloned_py_obj);
    for (size_t i = 0; i < obj_list.size(); ++i) {
      py::object each_cloned_obj = obj_list[i];
      (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_PARAMETER_FN_NAME, each_cloned_obj, py::str(phase),
                                     layout);
    }
  }
}

void InitCompileCacheParams(const pipeline::ResourcePtr &resource) {
  auto layout_dict = GetParameterLayoutFromResource(resource);
  auto graph = resource->func_graph();
  auto params = graph->parameters();
  for (auto &param : params) {
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (!param_ptr->has_default()) {
      continue;
    }
    SliceCacheParameterObj(param_ptr, layout_dict);
  }
}

void InitPynativeNoShardParams(const FuncGraphPtr &root) {
  auto parameters = root->parameters();
  for (auto &parameter : parameters) {
    auto param_ptr = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    auto param_info = param_ptr->param_info();
    if (!param_info) {
      MS_LOG(DEBUG) << "Parameter:" << parameter->DebugString() << " doesn't have param_info.";
      continue;
    }
    auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
    MS_EXCEPTION_IF_NULL(graph_executor);
    auto phase = graph_executor->phase();
    auto py_obj = GetPyParameterObj(param_info, OBJ);
    if (py::isinstance<py::none>(py_obj)) {
      MS_LOG(WARNING) << "Parameter: " << parameter->DebugString() << " can't find python obj.";
      continue;
    }
    (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, INIT_OPTIMIZER_STATE_FN, py_obj, py::str(phase));
  }
}

void AutoParallelPostProcess(const FuncGraphPtr &root) {
  auto parameters = root->parameters();
  for (auto &param : parameters) {
    if (ParameterIsCloned(param)) {
      continue;
    }
    auto layout = param->user_data<TensorLayout>();
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (!param_ptr->has_default()) {
      continue;
    }
    SliceParameterObj(param_ptr, layout);
  }
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
  auto py_instance = CreateOpInstance(attrs, "_VirtualAdd", "grad_accu");
  auto value_node = NewValueNode(py_instance);
  std::vector<AnfNodePtr> virtual_node_input = {value_node, cnode->input(IntToSize(node_user.second)), accu_parameter};
  auto graph = cnode->func_graph();
  auto virtual_node = graph->NewCNode(virtual_node_input);
  manager->SetEdge(cnode, node_user.second, virtual_node);
}

void HandleFullySplitParameters(const FuncGraphPtr &root) {
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  if ((grad_accumulation_step <= 1) || root->has_flag(kAccumulation)) {
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
  auto grad_accumulation_shard = ParallelContext::GetInstance()->grad_accumulation_shard();

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
        auto opt_shard_group = tensor_layout->opt_shard_group();
        auto opt_shard_shape = cloned_from_parameter->user_data<TensorLayout>()->opt_shard_slice_shape();
        std::shared_ptr<abstract::BaseShape> parallel_shape = nullptr;
        // set opt shard shape if the pipeline sharding is set
        if (grad_accumulation_shard && !opt_shard_group.empty()) {
          parallel_shape = std::make_shared<abstract::Shape>(opt_shard_shape);
        } else {
          parallel_shape = std::make_shared<abstract::Shape>(slice_shape);
        }
        MS_EXCEPTION_IF_NULL(parallel_shape);
        cloned_abstract->set_shape(parallel_shape);
        // in opt shard, accu_grad's shape is different from the original param's shape
        // if the grad_accumulation_shard is enabled, the accu_grads will be a opt-sharded shape
        if (!grad_accumulation_shard && ParallelContext::GetInstance()->enable_parallel_optimizer()) {
          TensorLayout new_layout = *tensor_layout;
          new_layout.set_opt_shard_group("");
          tensor_layout = std::make_shared<TensorLayout>(new_layout);
        }
      } else {
        cloned_abstract->set_shape(cloned_from_node->abstract()->GetShapeTrack());
      }
      cloned_parameter->set_user_data<TensorLayout>(tensor_layout);
      cloned_parameter_node->set_abstract(cloned_abstract);
      // copy the fusion tag
      auto cloned_param_info = cloned_parameter->param_info();
      MS_EXCEPTION_IF_NULL(cloned_param_info);
      auto cloned_from_param_info = cloned_from_parameter->param_info();
      MS_EXCEPTION_IF_NULL(cloned_from_param_info);
      cloned_param_info->set_comm_fusion(cloned_from_param_info->comm_fusion());

      MS_LOG(INFO) << "The parameter: " << cloned_parameter->name()
                   << " is cloned, the be cloned parameter is: " << cloned_from_parameter->name()
                   << ", clone index is:  " << cloned_index;
    } else {
      MS_LOG(EXCEPTION) << "The parameter: " << cloned_parameter->name() << " is cloned, cloned index is  "
                        << cloned_index << ", but not found the be cloned parameter";
    }
  }
}

// For adafactor optimizer, the relationship between parameter and state's shape as follows:
// 1) parameter: [A, B, C, D] (shape_size > 2), exp_avg_sq_row: [A, B, C], exp_avg_sq_col: [A, B, D], exp_avg_sq: [1]
//    If the parameter is opt shard, the exp_avg_sq_row and exp_avg_sq_col need to be shard accordingly.
// 2) parameter: [A, B] (shape_size = 2), exp_avg_sq_row: [A], exp_avg_sq_col: [B], exp_avg_sq: [1]
//    If the parameter is opt shard, the exp_avg_sq_row needs to be shard accordingly.
// 3) parameter: [A] (shape_size = 1), exp_avg_sq_row: [1], exp_avg_sq_col: [1], exp_avg_sq: [A]
//    If the parameter is opt shard, the exp_avg_sq needs to be shard accordingly.
static bool AdafactorStateIsOptShard(const std::string &opt_shard_group, size_t shape_size,
                                     const std::string &param_name, const std::string &state_name) {
  if (opt_shard_group.empty()) {
    return false;
  }

  std::string exp_row_name = EXP_AVG_SQ_ROW + param_name;
  std::string exp_col_name = EXP_AVG_SQ_COL + param_name;
  std::string exp_avg_name = EXP_AVG_SQ + param_name;

  if (shape_size > 2 && state_name == exp_avg_name) {
    return false;
  }

  if (shape_size == 2 && (state_name == exp_col_name || state_name == exp_avg_name)) {
    return false;
  }

  if (shape_size == 1 && (state_name == exp_row_name || state_name == exp_col_name)) {
    return false;
  }

  MS_LOG(INFO) << "The parameter " << param_name << " is opt shard";
  return true;
}

static bool IsOriginWeight(const ParameterPtr &param) {
  std::string param_name = param->name();
  if (param_name.find(EXP_AVG) != std::string::npos) {
    return false;
  }

  auto tensor_layout = param->user_data<TensorLayout>();
  if (tensor_layout == nullptr) {
    return false;
  }

  return true;
}

static std::pair<AnfNodePtr, bool> FindParameterByValueNode(const AnfNodePtr &node, const FuncGraphPtr &func_graph,
                                                            const std::string &name = ALL_REDUCE) {
  if (IsValueNode<RefKey>(node)) {
    std::vector<AnfNodePtr> param_v = FindParameterByRefKeyNode(node, func_graph);
    if (param_v.size() != 1) {
      MS_LOG(EXCEPTION) << "FindParameterByRefKeyNode failed, return vector size must be 1, real is  "
                        << param_v.size();
    }
    auto param_ptr = param_v[0]->user_data<parallel::TensorLayout>();
    if (param_ptr && !param_ptr->opt_shard_group().empty() && param_ptr->opt_shard_mirror_group().empty() &&
        name == ALL_REDUCE) {
      return std::make_pair(nullptr, true);
    }
    return std::make_pair(node, true);
  }
  return std::make_pair(nullptr, false);
}

static std::pair<AnfNodePtr, bool> FindParameterByParameter(const AnfNodePtr &node,
                                                            const std::string &name = ALL_REDUCE) {
  auto param_ptr = node->user_data<parallel::TensorLayout>();
  if (param_ptr && !param_ptr->opt_shard_group().empty() && param_ptr->opt_shard_mirror_group().empty() &&
      name == ALL_REDUCE) {
    return std::make_pair(nullptr, false);
  }
  return std::make_pair(node, false);
}

static std::pair<AnfNodePtr, bool> FindParameterByFuncGraph(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(fg);
  auto fg_parameters = fg->parameters();

  auto pre_node = GetRealKernelNode(fg->output(), -1, nullptr).first;
  auto pre_cnode = pre_node->cast<CNodePtr>();
  for (size_t index = 1; index < pre_cnode->inputs().size(); ++index) {
    auto res = FindParameter(pre_cnode->input(index), pre_cnode->func_graph());
    if (!res.first) {
      continue;
    }
    return res;
  }
  // If nothing found in the sub graphs, we search from the inputs of the graph.
  for (size_t index = 1; index < fg_parameters.size(); ++index) {
    auto res = FindParameter(cnode->input(index), fg);
    if (!res.first) {
      continue;
    }
    return res;
  }
  return std::make_pair(nullptr, false);
}

// Only used for InsertMirrorOps
std::pair<AnfNodePtr, bool> FindParameter(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  if (!node->isa<Parameter>() && !node->isa<CNode>() && !node->isa<ValueNode>()) {
    return std::make_pair(nullptr, false);
  }

  if (node->isa<Parameter>()) {
    return FindParameterByParameter(node);
  }

  if (node->isa<ValueNode>()) {
    return FindParameterByValueNode(node, func_graph);
  }

  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    for (size_t index = 0; index < cnode->inputs().size(); ++index) {
      auto res = FindParameter(cnode->input(index), func_graph);
      if (!res.first) {
        continue;
      }
      return res;
    }
  }

  // When not fully use opt shard, allgather and mirror would be both inserted.
  // Skip allgather here and find parameter recursively.
  if (IsParallelCareNode(cnode) && !IsInAllGatherNodeList(cnode)) {
    return std::make_pair(nullptr, false);
  }
  if (IsValueNode<FuncGraph>(cnode->input(0))) {
    return FindParameterByFuncGraph(node);
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(prim_anf_node);
  for (size_t index = 0; index < cnode->inputs().size(); ++index) {
    PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(prim);
    if ((prim->name() == DEPEND || prim->name() == LOAD || IsInAllGatherNodeList(cnode)) && index != 1) {
      continue;
    }
    auto res = FindParameter(cnode->input(index), func_graph);
    if (!res.first) {
      continue;
    }
    return res;
  }
  return std::make_pair(nullptr, false);
}

// Used for allgather and reducescatter
std::pair<AnfNodePtr, bool> FindParameterWithAllgather(const AnfNodePtr &node, const FuncGraphPtr &func_graph,
                                                       const std::string &name) {
  if (!node->isa<Parameter>() && !node->isa<CNode>() && !node->isa<ValueNode>()) {
    return std::make_pair(nullptr, false);
  }

  if (node->isa<Parameter>()) {
    return FindParameterByParameter(node, name);
  }

  if (node->isa<ValueNode>()) {
    return FindParameterByValueNode(node, func_graph, name);
  }

  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t index = 0; index < cnode->inputs().size(); ++index) {
    if (index != 1) {
      continue;
    }
    auto res = FindParameterWithAllgather(cnode->input(index), func_graph, name);
    if (!res.first) {
      continue;
    }
    return res;
  }
  return std::make_pair(nullptr, false);
}

std::unordered_map<std::string, std::shared_ptr<TensorLayout>> AdaSumParamTensorLayout(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  std::unordered_map<std::string, std::shared_ptr<TensorLayout>> adasum_param_map;
  for (auto &parameter_node : root->parameters()) {
    MS_EXCEPTION_IF_NULL(parameter_node);
    auto cloned_parameter = parameter_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(cloned_parameter);

    if (!ParameterIsCloned(parameter_node)) {
      auto parameter_tensor_layout = cloned_parameter->user_data<TensorLayout>();
      adasum_param_map["adasum_delta_weight." + cloned_parameter->name()] = parameter_tensor_layout;
    }
  }
  return adasum_param_map;
}

Shape ValueSequeueScaleToShape(const ValuePtr &value_seq, const Shape &scale, size_t expand_ratio = 1) {
  if (!value_seq->isa<ValueSequeue>()) {
    MS_LOG(EXCEPTION) << "The input is not a value_sequeue";
  }
  std::vector<int64_t> origin_value_vector;
  if (TransValueSequeueToVector(value_seq, &origin_value_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Transform value_seq to vector failed";
  }
  if (origin_value_vector.size() != scale.size()) {
    MS_LOG(EXCEPTION) << "Shape not equal, cannot scale, value_seq size is: " << origin_value_vector.size()
                      << " scale size is: " << scale.size();
  }
  for (size_t i = 0; i < scale.size(); ++i) {
    origin_value_vector[i] = origin_value_vector[i] / scale[i];
    if (i == 0) {
      origin_value_vector[i] = origin_value_vector[i] * SizeToLong(expand_ratio);
    }
  }
  return origin_value_vector;
}

ValuePtr ValueSequeueScale(const ValuePtr &value_seq, const Shape &scale, size_t expand_ratio = 1) {
  Shape origin_value_vector = ValueSequeueScaleToShape(value_seq, scale, expand_ratio);
  if (value_seq->isa<ValueTuple>()) {
    return TransVectorToValueSequeue<ValueTuple>(origin_value_vector);
  }
  return TransVectorToValueSequeue<ValueList>(origin_value_vector);
}

void ReplaceAdaSumStridedSliceValue(const CNodePtr &stridedslice_cnode1,
                                    const std::shared_ptr<TensorLayout> &target_param_layout,
                                    size_t slice_expand_ratio) {
  auto target_param_info = std::make_shared<TensorInfo>(target_param_layout->SqueezeShape());
  Dimensions param_strategy = target_param_info->InferStrategy();
  auto new_begin1_value =
    ValueSequeueScale(GetValueNode(stridedslice_cnode1->input(2)), param_strategy, slice_expand_ratio);
  auto new_end1_value =
    ValueSequeueScale(GetValueNode(stridedslice_cnode1->input(3)), param_strategy, slice_expand_ratio);
  ValueNodePtr new_begin_value_node = std::make_shared<ValueNode>(new_begin1_value);
  ValueNodePtr new_end_value_node = std::make_shared<ValueNode>(new_end1_value);
  stridedslice_cnode1->set_input(2, new_begin_value_node);
  stridedslice_cnode1->set_input(3, new_end_value_node);
}

RankList GetRankListByLayout(const std::shared_ptr<TensorLayout> &target_param_layout) {
  int64_t rank = g_device_manager->global_rank();
  auto dev_shape = target_param_layout->device_arrangement().array();
  auto stage_device_list = g_device_manager->GetDeviceListInThisStage();
  DeviceMatrix dev_matrix(rank, stage_device_list, dev_shape);
  RankList group_devices;
  if (dev_matrix.GetDevicesByTensorMap(target_param_layout->tensor_map().array(), &group_devices) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Get adasum parameter origin mirror group by tensor layout failed.";
  }
  return group_devices;
}

std::vector<bool> IsBorderAdaSumSendReceive(const AnfNodePtr &node, const RankList &group_devices) {
  bool is_send = IsPrimitiveCNode(node, prim::kPrimSend);
  PrimitivePtr send_rec_prim = GetCNodePrimitive(node);
  int64_t origin_dest_rank = GetValue<int64_t>(send_rec_prim->GetAttr(OPPOSITE_RANK));
  int64_t rank = g_device_manager->global_rank();
  if (group_devices.size() - 1 == 0) {
    MS_LOG(EXCEPTION) << "May division by zero.";
  }
  int64_t adasum_rank_distance = (group_devices.back() - group_devices.front()) / SizeToLong(group_devices.size() - 1);
  if (adasum_rank_distance < ADASUM_MIN_DIS) {
    adasum_rank_distance = ADASUM_MIN_DIS;
  }
  size_t border_step = size_t(log2(adasum_rank_distance / ADASUM_MIN_DIS));
  int64_t fusion_id = GetValue<int64_t>(send_rec_prim->GetAttr("origin_fusion"));
  // when cutting nodes, the fusion id should change.
  int64_t new_fusion_id = fusion_id + SizeToLong(g_device_manager->DeviceNum() * (border_step + IntToSize(1)));
  send_rec_prim->set_attr(FUSION, MakeValue(new_fusion_id));
  std::vector<int64_t> group_list;
  int64_t new_dest_src_rank;
  if (rank > origin_dest_rank) {
    group_list = {origin_dest_rank, rank};
    new_dest_src_rank = 0;
  } else {
    group_list = {rank, origin_dest_rank};
    new_dest_src_rank = 1;
  }
  Group adasum_send_rec_group;
  if (g_device_manager->CreateGroup(group_list, &adasum_send_rec_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create send/receive group in adasum failed, the group is:" << group_list;
  }
  send_rec_prim->set_attr(GROUP, MakeValue(adasum_send_rec_group.name()));
  if (is_send) {
    send_rec_prim->set_attr(DEST_RANK, MakeValue(new_dest_src_rank));
  } else {
    send_rec_prim->set_attr(SRC_RANK, MakeValue(new_dest_src_rank));
  }
  int64_t rank_dis = abs(origin_dest_rank - rank);
  if (adasum_rank_distance == ADASUM_MIN_DIS) {
    return {false, false, false, false};
  }
  bool is_origin_first_node_if_forward = false;
  bool is_new_first_node_if_forward = false;
  bool is_origin_last_node_if_rollback = false;
  bool is_new_last_node_if_rollback = false;
  if (rank_dis == ADASUM_MIN_DIS) {
    is_origin_first_node_if_forward = true;
    is_origin_last_node_if_rollback = true;
  }
  if (rank_dis == adasum_rank_distance) {
    is_new_first_node_if_forward = true;
  }
  if (rank_dis == adasum_rank_distance / 2) {
    is_new_last_node_if_rollback = true;
  }
  return {is_origin_first_node_if_forward, is_new_first_node_if_forward, is_origin_last_node_if_rollback,
          is_new_last_node_if_rollback};
}

void HandleAdaSumReshape(const CNodePtr &reshape_cnode, const std::shared_ptr<TensorLayout> &target_param_layout) {
  auto slice_shape = target_param_layout->slice_shape().array();
  auto slice_shape_value = TransVectorToValueSequeue<ValueTuple>(slice_shape);
  ValueNodePtr new_slice_shape_value_node = std::make_shared<ValueNode>(slice_shape_value);
  reshape_cnode->set_input(2, new_slice_shape_value_node);
}

void RemoveAdasumRedundantNodes(const FuncGraphManagerPtr &manager,
                                std::unordered_map<std::string, CNodePtr> *forward_origin_first_node_map,
                                std::unordered_map<std::string, CNodePtr> *forward_new_first_node_map,
                                std::unordered_map<std::string, CNodePtr> *rollback_origin_last_node_map,
                                std::unordered_map<std::string, CNodePtr> *rollback_new_last_node_map) {
  // connect forward last node and rollback first node
  if (forward_origin_first_node_map->size() != forward_new_first_node_map->size() ||
      rollback_origin_last_node_map->size() != rollback_new_last_node_map->size()) {
    MS_LOG(EXCEPTION) << "The over border node is not equal in adasum forward process and rollback process.";
  }
  for (auto node : *forward_origin_first_node_map) {
    std::string target_param = node.first;
    CNodePtr forward_origin_first_node = node.second;
    CNodePtr forward_new_first_node = (*forward_new_first_node_map)[target_param];
    manager->SetEdge(forward_new_first_node, 1, forward_origin_first_node->input(1));
  }
  for (auto node : *rollback_origin_last_node_map) {
    std::string target_param = node.first;
    CNodePtr rollback_origin_last_node = node.second;
    CNodePtr rollback_new_last_node = (*rollback_new_last_node_map)[target_param];
    (void)manager->Replace(rollback_origin_last_node, rollback_new_last_node);
  }
}

void HandleAdasumAllReduce(const PrimitivePtr &prim, const RankList &group_devices) {
  size_t step = size_t(GetValue<int64_t>(prim->GetAttr("step")));
  std::vector<int64_t> neighbor_ids;
  int64_t adasum_rank_distance =
    (group_devices.back() - group_devices.front()) / SizeToLong((group_devices.size() - 1));
  if (adasum_rank_distance < ADASUM_MIN_DIS) {
    adasum_rank_distance = ADASUM_MIN_DIS;
  }
  size_t border_step = size_t(log2(adasum_rank_distance / ADASUM_MIN_DIS));
  MS_LOG(INFO) << "current border step is: " << border_step;
  if (step < border_step) {
    return;
  }
  int64_t rank = g_device_manager->global_rank();
  size_t double_d = size_t(IntToSize(2) << step);
  for (size_t index = 0; index < double_d; ++index) {
    int64_t node_rank = rank / ADASUM_MIN_DIS;
    int64_t neighbor_id =
      (node_rank / SizeToLong(double_d) * SizeToLong(double_d) + SizeToLong(index)) * ADASUM_MIN_DIS +
      rank % ADASUM_MIN_DIS;
    neighbor_ids.push_back(neighbor_id);
  }
  Group adasum_allreduce_group;
  if (g_device_manager->CreateGroup(neighbor_ids, &adasum_allreduce_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create group allreduce group in adasum failed, the group is " << neighbor_ids;
  }
  auto new_group_name = MakeValue(adasum_allreduce_group.name());
  int64_t fusion_id = GetValue<int64_t>(prim->GetAttr("origin_fusion"));
  int64_t new_fusion_id = fusion_id + SizeToLong(g_device_manager->DeviceNum() * (border_step + IntToSize(1)));
  prim->set_attr(GROUP, new_group_name);
  prim->set_attr(FUSION, MakeValue(new_fusion_id));
}

void HandleAdasumSlice(const AnfNodePtr &stridedslice_node1, const std::shared_ptr<TensorLayout> &target_param_layout,
                       size_t slice_expand_ratio) {
  auto stridedslice_cnode1 = stridedslice_node1->cast<CNodePtr>();
  ReplaceAdaSumStridedSliceValue(stridedslice_cnode1, target_param_layout, slice_expand_ratio);
  auto squeeze_node = RealInputNode(stridedslice_cnode1, 1);
  if (!IsPrimitiveCNode(squeeze_node, prim::kPrimSqueeze)) {
    MS_LOG(EXCEPTION) << "The stridedslice input node should be squeeze in adasum";
  }
  auto squeeze_cnode = squeeze_node->cast<CNodePtr>();
  FuncGraphManagerPtr manager = squeeze_node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[squeeze_cnode];
  for (auto &node_pair : node_set) {
    if (IsPrimitiveCNode(node_pair.first, prim::kPrimStridedSlice) && node_pair.first != stridedslice_node1) {
      CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
      ReplaceAdaSumStridedSliceValue(use_apply, target_param_layout, slice_expand_ratio);
    }
  }
}

void HandleAdaSumConcat(const AnfNodePtr &concat_node, const std::vector<bool> &border_info,
                        const std::string &target_param,
                        std::unordered_map<std::string, CNodePtr> *rollback_new_last_node_map,
                        std::unordered_map<std::string, CNodePtr> *rollback_origin_last_node_map) {
  if (border_info[3]) {
    (*rollback_new_last_node_map)[target_param] = concat_node->cast<CNodePtr>();
  }
  if (border_info[2]) {
    auto manager = concat_node->func_graph()->manager();
    AnfNodeIndexSet concat_node_user_set = manager->node_users()[concat_node];
    for (auto &node_pair : concat_node_user_set) {
      if (IsPrimitiveCNode(node_pair.first, prim::kPrimMakeTuple)) {
        AnfNodeIndexSet make_tuple_node_user_set = manager->node_users()[node_pair.first];
        for (auto &tuple_user : make_tuple_node_user_set) {
          if (IsPrimitiveCNode(tuple_user.first, prim::kPrimConcat)) {
            (*rollback_origin_last_node_map)[target_param] = tuple_user.first->cast<CNodePtr>();
            return;
          }
        }
        return;
      }
    }
  }
}

void HandleAdaSumSqueeze(const AnfNodePtr &stridedslice_node1, const std::vector<bool> &border_info,
                         const std::string &target_param,
                         std::unordered_map<std::string, CNodePtr> *forward_origin_first_node_map,
                         std::unordered_map<std::string, CNodePtr> *forward_new_first_node_map) {
  auto squeeze_node = RealInputNode(stridedslice_node1->cast<CNodePtr>(), 1);
  if (border_info[0]) {
    (*forward_origin_first_node_map)[target_param] = squeeze_node->cast<CNodePtr>();
  }
  if (border_info[1]) {
    (*forward_new_first_node_map)[target_param] = squeeze_node->cast<CNodePtr>();
  }
}

void HandleAdaSumPureModelParallel(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
    return;
  }
  PrimitivePtr send_rec_prim = GetCNodePrimitive(node);
  int64_t origin_dest_rank = GetValue<int64_t>(send_rec_prim->GetAttr(OPPOSITE_RANK));
  int64_t rank = g_device_manager->global_rank();
  CNodePtr cnode = node->cast<CNodePtr>();
  auto pre_cnode = RealInputNode(cnode, 1);
  int64_t rank_dis = abs(origin_dest_rank - rank);
  if (rank_dis == ADASUM_MIN_DIS && IsPrimitiveCNode(pre_cnode, prim::kPrimStridedSlice)) {
    auto squeeze_node = pre_cnode->cast<CNodePtr>()->input(1);
    if (!IsPrimitiveCNode(squeeze_node, prim::kPrimSqueeze)) {
      return;
    }
    auto squeeze_input = squeeze_node->cast<CNodePtr>()->input(1);
    auto manager = squeeze_node->func_graph()->manager();
    AnfNodeIndexSet squeeze_input_node_user_set = manager->node_users()[squeeze_input];
    for (auto &squeeze_input_user : squeeze_input_node_user_set) {
      if (IsPrimitiveCNode(squeeze_input_user.first, prim::kPrimSqueeze) ||
          IsPrimitiveCNode(squeeze_input_user.first, prim::kPrimUpdateState) ||
          IsPrimitiveCNode(squeeze_input_user.first, prim::kPrimMakeTuple)) {
        continue;
      }
      (void)manager->Replace(squeeze_input_user.first, squeeze_input);
    }
  }
}

bool HandleAdaSum(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                  std::unordered_map<std::string, std::shared_ptr<TensorLayout>> *adasum_param_tensor_layout_map) {
  std::unordered_map<std::string, CNodePtr> forward_origin_first_node_map;
  std::unordered_map<std::string, CNodePtr> forward_new_first_node_map;
  std::unordered_map<std::string, CNodePtr> rollback_origin_last_node_map;
  std::unordered_map<std::string, CNodePtr> rollback_new_last_node_map;
  bool is_adasum = false;
  for (auto &node : all_nodes) {
    bool is_allreduce = IsPrimitiveCNode(node, prim::kPrimAllReduce);
    bool is_reshape = IsPrimitiveCNode(node, prim::kPrimReshape);
    bool is_send = IsPrimitiveCNode(node, prim::kPrimSend);
    bool is_receive = IsPrimitiveCNode(node, prim::kPrimReceive);
    if (!is_allreduce && !is_reshape && !is_send && !is_receive) {
      continue;
    }
    std::string target_param;
    CNodePtr cnode = node->cast<CNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0)->cast<ValueNodePtr>());
    if (!prim->HasAttr(TARGET_PARAM)) {
      continue;
    }
    target_param = GetValue<std::string>(prim->GetAttr(TARGET_PARAM));
    auto target_param_layout = (*adasum_param_tensor_layout_map)[target_param];
    RankList group_devices = GetRankListByLayout(target_param_layout);
    // only model parallel
    if (group_devices.size() == 1) {
      HandleAdaSumPureModelParallel(node);
      continue;
    }

    int64_t adasum_rank_distance =
      (group_devices.back() - group_devices.front()) / SizeToLong((group_devices.size() - 1));
    // when the repeat dim is right, the parameter do not enable adasum.
    if (adasum_rank_distance == 1 && group_devices.size() < size_t(g_device_manager->stage_device_num())) {
      continue;
    }
    MS_LOG(INFO) << "Apply adasum in auto parallel, current dealing node is: " << node->fullname_with_scope();
    is_adasum = true;
    size_t slice_expand_ratio =
      LongToSize(adasum_rank_distance / ADASUM_MIN_DIS) > 0 ? LongToSize(adasum_rank_distance / ADASUM_MIN_DIS) : 1;
    if (is_reshape) {
      HandleAdaSumReshape(cnode, (*adasum_param_tensor_layout_map)[target_param]);
    }
    if (is_allreduce && prim->HasAttr("step")) {
      HandleAdasumAllReduce(prim, group_devices);
    }
    if (is_send || is_receive) {
      std::vector<bool> border_info = IsBorderAdaSumSendReceive(node, group_devices);
      if (is_receive) {
        auto target_param_info = std::make_shared<TensorInfo>(*target_param_layout);
        Dimensions param_strategy = target_param_info->InferStrategy();
        Shape new_rec_shape = ValueSequeueScaleToShape(prim->GetAttr(SHAPE), param_strategy, slice_expand_ratio);
        auto new_rec_shape_value = TransVectorToValueSequeue<ValueList>(new_rec_shape);
        prim->set_attr(SHAPE, new_rec_shape_value);
        continue;
      }
      auto stridedslice_node1 = RealInputNode(cnode, 1);
      if (IsPrimitiveCNode(stridedslice_node1, prim::kPrimConcat)) {
        HandleAdaSumConcat(stridedslice_node1, border_info, target_param, &rollback_new_last_node_map,
                           &rollback_origin_last_node_map);
        continue;
      }
      if (!IsPrimitiveCNode(stridedslice_node1, prim::kPrimStridedSlice)) {
        continue;
      }
      HandleAdasumSlice(stridedslice_node1, target_param_layout, slice_expand_ratio);
      HandleAdaSumSqueeze(stridedslice_node1, border_info, target_param, &forward_origin_first_node_map,
                          &forward_new_first_node_map);
    }
  }
  RemoveAdasumRedundantNodes(root->manager(), &forward_origin_first_node_map, &forward_new_first_node_map,
                             &rollback_origin_last_node_map, &rollback_new_last_node_map);
  return is_adasum;
}

void ResetMirrorAttr(const PrimitivePtr &prim, const RankList &new_group) {
  if (new_group.size() == 1) {
    prim->set_attr(DEV_NUM, MakeValue<int64_t>(SizeToLong(new_group.size())));
    prim->set_attr(GROUP, MakeValue("one_rank_group"));
    prim->set_attr(GROUP_RANKS, MakeValue(std::to_string(new_group[0])));
    return;
  }
  Group adasum_mirror_group;
  if (g_device_manager->CreateGroup(new_group, &adasum_mirror_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create new mirror group failed in adasum, new group is: " << new_group;
  }
  auto new_group_name = MakeValue(adasum_mirror_group.name());
  prim->set_attr(GROUP, new_group_name);
  prim->set_attr(DEV_NUM, MakeValue<int64_t>(SizeToLong(new_group.size())));
  std::string rank_list_name = g_device_manager->FindRankListNameByHashName(adasum_mirror_group.name());
  prim->set_attr(GROUP_RANKS, MakeValue(rank_list_name));
}

void HandleMirrorInAdaSum(
  const FuncGraphPtr &root,
  std::unordered_map<std::string, std::shared_ptr<TensorLayout>> *adasum_param_tensor_layout_map) {
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(root->get_return());
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimMirror)) {
      continue;
    }
    CNodePtr mirror_cnode = node->cast<CNodePtr>();
    auto param_node_pair = FindParameter(mirror_cnode->input(1), node->func_graph());
    if (!param_node_pair.first) {
      MS_LOG(EXCEPTION) << "Mirror input is not a param";
    }
    auto param_ptr = param_node_pair.first->cast<ParameterPtr>();
    std::string param_name = param_ptr->name();
    MS_LOG(INFO) << "Mirror param name is: " << param_name;
    std::string target_param = "adasum_delta_weight." + param_name;
    auto target_param_layout = (*adasum_param_tensor_layout_map)[target_param];

    // Change mirror group
    RankList group_devices = GetRankListByLayout(target_param_layout);
    int64_t rank = g_device_manager->global_rank();
    size_t group_dis = LongToSize(group_devices.back() - group_devices.front()) / (group_devices.size() - 1);
    auto prim = GetCNodePrimitive(node);
    if (group_dis < ADASUM_MIN_DIS && group_dis > 0) {
      size_t new_group_size = size_t(ADASUM_MIN_DIS) / group_dis;
      // compute new group range
      size_t group_begin = 0;
      for (size_t group_end = new_group_size; group_end < group_devices.size() + new_group_size;
           group_end += new_group_size) {
        int64_t max_group_value =
          group_end >= group_devices.size() ? (group_devices.back() + 1) : group_devices[group_end];
        if (group_devices[group_begin] <= rank && rank < max_group_value) {
          std::vector<int64_t> new_group(group_devices.begin() + SizeToLong(group_begin),
                                         group_devices.begin() + SizeToLong(group_end));
          MS_LOG(INFO) << "Find new mirror group in adasum: " << new_group << " target_param:" << target_param;
          ResetMirrorAttr(prim, new_group);
          break;
        }
        group_begin = group_end;
      }
      continue;
    }
    ResetMirrorAttr(prim, {rank});
  }
}

void HandleAdaFactorOpt(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  for (auto &param_node : root->parameters()) {
    MS_EXCEPTION_IF_NULL(param_node);
    auto param = param_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);

    if (!IsOriginWeight(param)) {
      continue;
    }

    int64_t row_col_count = 0;
    int64_t exp_avg_sq_count = 0;
    for (auto &row_col_node : root->parameters()) {
      if (row_col_count == 2 && exp_avg_sq_count == 1) {
        break;
      }

      MS_EXCEPTION_IF_NULL(row_col_node);
      auto row_col_param = row_col_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(row_col_param);
      std::string row_col_param_name = row_col_param->name();
      std::string param_name = param->name();
      std::string exp_row_name = EXP_AVG_SQ_ROW + param_name;
      std::string exp_col_name = EXP_AVG_SQ_COL + param_name;
      std::string exp_avg_name = EXP_AVG_SQ + param_name;

      if ((row_col_param_name != exp_row_name) && (row_col_param_name != exp_col_name) &&
          (row_col_param_name != exp_avg_name)) {
        continue;
      }

      auto tensor_layout = param->user_data<TensorLayout>();
      MS_EXCEPTION_IF_NULL(tensor_layout);
      auto slice_shape = tensor_layout->slice_shape().array();
      Shape opt_shard_slice_shape = slice_shape;
      if (!tensor_layout->opt_shard_group().empty()) {
        opt_shard_slice_shape = tensor_layout->opt_shard_slice_shape();
      }

      auto shape_size = slice_shape.size();
      bool is_row_or_col_param = (row_col_param_name == exp_row_name) || (row_col_param_name == exp_col_name);
      if (is_row_or_col_param && shape_size <= 1) {
        row_col_count++;
        continue;
      }

      if (row_col_param_name == exp_avg_name && shape_size != 1) {
        exp_avg_sq_count++;
        continue;
      }

      auto origin_shape = tensor_layout->tensor_shape().array();
      auto dev_mat = tensor_layout->device_arrangement().array();
      auto tensor_map = tensor_layout->tensor_map().array();

      if (row_col_param_name == exp_row_name) {
        opt_shard_slice_shape.pop_back();
        origin_shape.pop_back();
        tensor_map.pop_back();
        row_col_count++;
      } else if (row_col_param_name == exp_col_name) {
        (void)opt_shard_slice_shape.erase(opt_shard_slice_shape.cbegin() +
                                          static_cast<different_type>(SECOND_FROM_END(shape_size)));
        (void)origin_shape.erase(origin_shape.cbegin() + static_cast<different_type>(SECOND_FROM_END(shape_size)));
        (void)tensor_map.erase(tensor_map.cbegin() + static_cast<different_type>(SECOND_FROM_END(shape_size)));
        row_col_count++;
      } else {
        exp_avg_sq_count++;
      }

      TensorLayout new_tensor_layout;
      if (new_tensor_layout.InitFromVector(dev_mat, tensor_map, origin_shape) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Init tensor layout failed";
      }

      if (AdafactorStateIsOptShard(tensor_layout->opt_shard_group(), shape_size, param_name, row_col_param_name)) {
        new_tensor_layout.set_opt_shard_group(tensor_layout->opt_shard_group());
      }

      auto cloned_abstract = row_col_node->abstract()->Clone();
      MS_EXCEPTION_IF_NULL(cloned_abstract);
      std::shared_ptr<abstract::BaseShape> parallel_shape = std::make_shared<abstract::Shape>(opt_shard_slice_shape);
      MS_EXCEPTION_IF_NULL(parallel_shape);
      cloned_abstract->set_shape(parallel_shape);
      row_col_param->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(new_tensor_layout));
      row_col_node->set_abstract(cloned_abstract);
      MS_LOG(INFO) << "Set the slice shape for " << row_col_param_name << ", origin shape is " << origin_shape
                   << ", new slice shape is " << opt_shard_slice_shape;
    }
  }
}

static std::shared_ptr<TensorLayout> FindParameterNextLayout(const AnfNodePtr &node, size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When finding the next tensor layout for the parameter, exceeded the maximum recursion depth: "
                    << MAX_RECURSIVE_DEPTH;
    return nullptr;
  }
  FuncGraphManagerPtr manager = node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[node];
  for (auto &node_pair : node_set) {
    if (IsPrimitiveCNode(node_pair.first, prim::kPrimLoad)) {
      auto layout_param = FindParameterNextLayout(node_pair.first, ++curr_depth);
      if (!layout_param) {
        continue;
      }
      return layout_param;
    }
    CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = use_apply->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if ((node_prim->name() == DEPEND && node_pair.second != 1) || node_prim->name() == RESHAPE) {
      continue;
    }
    if (IsParallelCareNode(use_apply) && use_apply->has_user_data<OperatorInfo>()) {
      auto layout = GetInputLayoutFromCNode(node_pair);
      return std::make_shared<TensorLayout>(layout);
    }
  }
  return nullptr;
}

std::shared_ptr<TensorLayout> CreateParameterLayout(const AnfNodePtr &node) {
  // Create DataParallel tensor layout for parameter(support WideDeep).
  auto next_layout = FindParameterNextLayout(node, 0);
  if (next_layout != nullptr) {
    return next_layout;
  }
  CheckGlobalDeviceManager();
  int64_t dev_num = g_device_manager->stage_device_num();
  TensorLayout input_tensor_layout;
  // create input_shape
  Shapes inputs_shape = GetNodeShape(node);
  Shape input_shape_array = inputs_shape[0];
  if (input_shape_array.empty()) {
    MS_LOG(EXCEPTION) << "Don't support reshape a scalar parameter.";
  }
  // create tensor_map
  size_t shape_size = input_shape_array.size();
  TensorMap input_tensor_map_array(SizeToLong(shape_size) - 1, -1);
  (void)input_tensor_map_array.insert(input_tensor_map_array.cbegin(), 0);
  // create dev_matrix
  Shape dev_matrix_array = {dev_num};
  if (input_tensor_layout.InitFromVector(dev_matrix_array, input_tensor_map_array, input_shape_array) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create tensor layout for parameter failed.";
  }
  return std::make_shared<TensorLayout>(input_tensor_layout);
}
}  // namespace parallel
}  // namespace mindspore
