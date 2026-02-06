/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "frontend/jit/ps/static_analysis/parameter_forward_hook.h"
#include "frontend/operator/ops.h"
#include "frontend/jit/ps/parse/parse.h"
#include "ir/func_graph.h"
#include "ir/func_graph_flag.h"
#include "ir/value.h"
#include "include/frontend/jit/ps/parse/py_data_convert.h"
#include "include/utils/python_adapter.h"
#include "include/utils/tensor_py_wrapper.h"
#include "pybind11/pybind11.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace abstract {

constexpr auto kParameterForwardHookAttr = "__parameter_forward_hook__";

void ClearParameterForwardHookStack() { ParameterForwardHookStack::Instance().clear(); }

bool IsHookedParameterCNode(const AnfNodePtr &anf_node) {
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }
  auto param_fwd_hook = cnode->GetAttr(kParameterForwardHook);
  if (param_fwd_hook == nullptr) {
    return false;
  }
  return GetValue<bool>(param_fwd_hook);
}

py::object GetTopParameterForwardHookFrame() {
  const auto &hook = abstract::ParameterForwardHookStack::Instance();
  if (hook.empty()) {
    return py::none();
  }

  auto top_frame = hook.top();
  if (!top_frame.second) {
    return py::none();
  }

  auto top_frame_fg = top_frame.first->func_graph();
  MS_EXCEPTION_IF_NULL(top_frame_fg);
  auto top_frame_obj_value = top_frame_fg->python_obj();
  MS_EXCEPTION_IF_NULL(top_frame_obj_value);
  auto top_frame_wrapper = dyn_cast_ptr<parse::PyObjectWrapper>(top_frame_obj_value);
  MS_EXCEPTION_IF_NULL(top_frame_wrapper);
  return top_frame_wrapper->obj();
}

AnfNodePtr ApplyParameterForwardHook(const FuncGraphPtr &func_graph, const ParameterPtr &param_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(param_node);
  constexpr auto kHookedParameter = "__hooked_parameter__";

  if (param_node->has_user_data(kHookedParameter)) {
    return param_node->user_data<AnfNode>(kHookedParameter);
  }

  if (!param_node->has_default()) {
    MS_LOG(EXCEPTION) << "Parameter must have a default value: " << param_node->ToString();
  }
  auto default_param = param_node->default_param_raw()->cast<tensor::TensorPyWrapperPtr>();
  MS_EXCEPTION_IF_NULL(default_param);
  auto param_obj = default_param->GetTensorWrapper();
  py::object hook_source_obj = param_obj;

  auto fwd_hook = python_adapter::GetPyObjAttr(param_obj, kParameterForwardHookAttr);
  if (py::isinstance<py::none>(fwd_hook)) {
    auto top_frame_obj = GetTopParameterForwardHookFrame();
    fwd_hook = python_adapter::GetPyObjAttr(top_frame_obj, kParameterForwardHookAttr);
    if (py::isinstance<py::none>(fwd_hook)) {
      return param_node;
    }
    hook_source_obj = top_frame_obj;
  }

  MS_LOG(DEBUG) << "Apply parameter-forward-hook for parameter: " << param_node->ToString() << " from `"
                << std::string(py::str(hook_source_obj)) << "`.";

  ValuePtr fwd_hook_fg = nullptr;
  bool succ = parse::ConvertData(fwd_hook, &fwd_hook_fg);
  if (!succ) {
    MS_LOG(EXCEPTION) << "Failed to convert fwd_hook to ValuePtr for parameter: " << param_node->ToString();
  }

  CNodePtr fwd_hook_node = func_graph->NewCNodeInFront({NewValueNode(fwd_hook_fg), param_node});
  fwd_hook_node->AddAttr(kParameterForwardHook, MakeValue(true));
  param_node->set_user_data<AnfNode>(kHookedParameter, fwd_hook_node);

  return fwd_hook_node;
}

bool ShouldSkipParameterForwardHook(const AnfNodePtr &origin_node, const ParameterPtr &para_node) {
  if (origin_node == nullptr) {
    return false;
  }
  auto cnode = origin_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }

  auto attr = cnode->GetAttr(kFuncGraphFlagNoParameterForwardHook);
  if (attr != nullptr) {
    return GetValue<bool>(attr);
  }

  // Check if we are currently inside a enable "parameter-forward hook" context.
  // If not, mark the current CNode to no parameter forward hook parsing.
  auto &hook = abstract::ParameterForwardHookStack::Instance();
  if (hook.empty()) {
    return true;
  }

  const auto &[top_frame_fg, top_frame_active] = hook.top();
  MS_EXCEPTION_IF_NULL(top_frame_fg);
  MS_LOG(DEBUG) << (top_frame_active ? "enable" : "disable")
                << " parameter-forward-hook for parameter: " << para_node->DebugString()
                << " in CNode: " << cnode->DebugString()
                << " of parameter-forward-hook context: " << top_frame_fg->ToString();
  return !top_frame_active;
}

void ReOrderParameterForwardHookNodes(const FuncGraphPtr &new_fg, const pipeline::ResourcePtr &resource) {
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &all_nodes = manager->all_nodes();
  auto &node_users = manager->node_users();
  std::map<CNodePtr, AnfNodeIndexSet> todos;

  for (const auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto param_fwd_hook = cnode->GetAttr(kParameterForwardHook);
    if (param_fwd_hook == nullptr) {
      continue;
    }

    if (!GetValue<bool>(param_fwd_hook)) {
      continue;
    }

    auto iter = node_users.find(node);
    if (iter == node_users.end()) {
      continue;
    }
    todos[cnode] = iter->second;
  }

  for (const auto &cnode_and_users : todos) {
    auto param = cnode_and_users.first->input(1);
    auto param_node = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);
    if (!param_node->has_default()) {
      continue;
    }

    auto reorder_hooked_cnode = [&manager](const CNodePtr &cnode, const AnfNodeIndexSet &users) {
      auto node_fg = cnode->func_graph();
      MS_EXCEPTION_IF_NULL(node_fg);

      const auto &order_list = node_fg->order_list();
      auto iter = std::find_if(order_list.begin(), order_list.end(), [&cnode](const CNodeWeakPtr &node) {
        auto cnode_in_order = node.lock();
        if (cnode_in_order == nullptr) {
          return false;
        }
        return std::any_of(cnode_in_order->inputs().begin(), cnode_in_order->inputs().end(),
                           [&cnode](const auto &input) { return input != nullptr && input == cnode; });
      });

      CNodePtr new_cnode = nullptr;
      if (iter == order_list.end()) {
        new_cnode = node_fg->NewCNodeInFront(cnode->inputs());
      } else {
        auto first_user = (*iter).lock();
        new_cnode = node_fg->NewCNodeBefore(first_user, cnode->inputs());
      }
      new_cnode->CloneCNodeInfo(cnode);
      for (const auto &[user, idx] : users) {
        manager->SetEdge(user, idx, new_cnode);
      }
      node_fg->EraseUnusedNodeInOrder(cnode);
    };

    reorder_hooked_cnode(cnode_and_users.first, cnode_and_users.second);
  }
}
}  // namespace abstract
}  // namespace mindspore
