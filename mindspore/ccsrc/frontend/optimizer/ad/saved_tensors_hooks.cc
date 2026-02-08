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

#include "frontend/optimizer/ad/saved_tensors_hooks.h"

#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "ir/anf.h"
#include "ir/core_ops_primitive.h"
#include "ir/func_graph.h"
#include "ir/func_graph_flag.h"
#include "ir/manager.h"
#include "ir/graph_utils.h"
#include "frontend/jit/ps/parse/resolve.h"

namespace mindspore {
namespace ad {

const char kUserDataAbs[] = "user_data_abs";

constexpr size_t kCNodeFirstDataInput = 1;
constexpr size_t kCNodeSecondDataInput = 2;
constexpr size_t kKGraphOutPutCnodeSize = 3;

SavedTensorsHooks::Stack &SavedTensorsHooks::Stack::GetInstance() {
  static SavedTensorsHooks::Stack instance;
  return instance;
}

void SavedTensorsHooks::Stack::Enter(const FuncGraphPtr &pack_hook, const FuncGraphPtr &unpack_hook) {
  MS_LOG(DEBUG) << "Entering saved tensors hooks: pack hook =" << pack_hook->ToString()
                << ", unpack hook =" << unpack_hook->ToString();
  stk_.push({pack_hook, unpack_hook});
}

void SavedTensorsHooks::Stack::Exit() {
  if (stk_.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Unbalanced push/pop for saved tensors hooks";
  }

  MS_LOG(DEBUG) << "Exiting saved tensors hooks: pack=" << (stk_.top().first)->ToString()
                << ", unpack=" << (stk_.top().second)->ToString();

  stk_.pop();
}

const FuncGraphPtr SavedTensorsHooks::Stack::pack_hook() const { return stk_.empty() ? nullptr : stk_.top().first; }
const FuncGraphPtr SavedTensorsHooks::Stack::unpack_hook() const { return stk_.empty() ? nullptr : stk_.top().second; }

SavedTensorsHooks::SavedTensorsHooks(const FuncGraphPtr &func_graph) {
  auto python_obj = func_graph->python_obj();
  if (python_obj == nullptr) {
    return;
  }

  auto py_obj_wrapper = python_obj->cast<parse::PyObjectWrapperPtr>();
  if (py_obj_wrapper == nullptr) {
    return;
  }
  auto obj = py_obj_wrapper->obj();

  auto pack_hook_obj = py::getattr(obj, FUNC_GRAPH_FLAG_PACK_HOOK, py::none());
  auto unpack_hook_obj = py::getattr(obj, FUNC_GRAPH_FLAG_UNPACK_HOOK, py::none());
  if (pack_hook_obj.is_none() && unpack_hook_obj.is_none()) {
    py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    auto construct_obj = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_ORIGINAL_CELL_CONSTRUCT, obj);
    if (construct_obj.is_none()) {
      return;
    }
    obj = construct_obj;
  }

  auto pack_hook = parse::ResolveSaveTensorHook(obj, FUNC_GRAPH_FLAG_PACK_HOOK);
  auto unpack_hook = parse::ResolveSaveTensorHook(obj, FUNC_GRAPH_FLAG_UNPACK_HOOK);
  if (pack_hook == nullptr && unpack_hook == nullptr) {
    return;
  }

  auto make_id_func_graph = [](const std::string &name) {
    auto fn = std::make_shared<FuncGraph>();
    fn->debug_info()->set_name(name);
    fn->set_output(fn->add_parameter());
    return fn;
  };

  // Make sure pack_hook and unpack_hook are in pair.
  pack_hook = (pack_hook == nullptr) ? make_id_func_graph("identity_pack_hook") : pack_hook;
  unpack_hook = (unpack_hook == nullptr) ? make_id_func_graph("identity_unpack_hook") : unpack_hook;

  auto set_flags_and_manager = [&func_graph](const FuncGraphPtr &fg) {
    fg->set_manager(func_graph->manager());
    fg->set_flag(mindspore::kFuncGraphFlagBackPropEntry, true);
    fg->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
  };

  set_flags_and_manager(pack_hook);
  set_flags_and_manager(unpack_hook);

  has_saved_tensors_hooks_ = true;
  SavedTensorsHooks::Stack::GetInstance().Enter(pack_hook, unpack_hook);
}

SavedTensorsHooks::~SavedTensorsHooks() {
  if (has_saved_tensors_hooks_) {
    SavedTensorsHooks::Stack::GetInstance().Exit();
  }
}

std::map<AnfNodePtr, std::set<std::pair<CNodePtr, size_t>>> UsersOfNeedSavedTensors(const FuncGraphPtr &bprop_env_fg,
                                                                                    const FuncGraphPtr &k) {
  MS_EXCEPTION_IF_NULL(bprop_env_fg);
  MS_EXCEPTION_IF_NULL(k);

  std::map<AnfNodePtr, std::set<std::pair<CNodePtr, size_t>>> users_of_need_saved_tensors;

  (void)TopoSort(bprop_env_fg->get_return(), SuccDeeperSimple, [&users_of_need_saved_tensors, &k](const AnfNodePtr &n) {
    if (n->func_graph() == k) {
      return mindspore::IncludeType::EXCLUDE;
    }

    auto cnode = n->cast<CNodePtr>();
    if (cnode == nullptr) {
      return mindspore::IncludeType::FOLLOW;
    }

    for (auto i = kCNodeFirstDataInput; i < cnode->size(); ++i) {
      auto &input = cnode->input(i);
      if (input->func_graph() == k && !mindspore::IsMonad(input)) {
        auto abs = input->user_data<abstract::AbstractBase>(kUserDataAbs);
        // Only tensor is saved
        if (abs == nullptr || !abs->isa<abstract::AbstractUndetermined>()) {
          continue;
        }

        // Only tensor with kValueAny is saved.
        auto value = abs->BuildValue();
        if (value == nullptr || value != kValueAny) {
          continue;
        }
        users_of_need_saved_tensors[input].emplace(cnode, i);
      }
    }

    return mindspore::IncludeType::FOLLOW;
  });

  return users_of_need_saved_tensors;
}

void SetUserDataAbs(const std::vector<AnfNodePtr> &transf_args, const AnfNodePtr &out_value, const CNodePtr &cnode,
                    const FuncGraphPtr &current_primal_fg) {
  if (cnode != nullptr) {
    MS_EXCEPTION_IF_CHECK_FAIL(transf_args.size() <= cnode->size(),
                               "Generated args count exceeds original CNode input count.");
    for (size_t idx = 1; idx < cnode->size(); ++idx) {
      transf_args[idx]->set_user_data(kUserDataAbs, cnode->input(idx)->abstract());
    }
    out_value->set_user_data(kUserDataAbs, cnode->abstract());
  } else {
    if (current_primal_fg == nullptr) {
      return;
    }

    MS_EXCEPTION_IF_CHECK_FAIL(transf_args.size() <= current_primal_fg->parameters().size() + 1,
                               "Generated args count exceeds expected count based on primal FuncGraph parameters.");
    for (size_t idx = 0; idx < current_primal_fg->parameters().size(); ++idx) {
      transf_args[idx + 1]->set_user_data(kUserDataAbs, current_primal_fg->parameters()[idx]->abstract());
    }
    MS_EXCEPTION_IF_NULL(current_primal_fg->output());
    out_value->set_user_data(kUserDataAbs, current_primal_fg->output()->abstract());
  }
}

void ClearUserDataAbs(const std::vector<AnfNodePtr> &transf_args, const AnfNodePtr &out_value) {
  for (auto &arg : transf_args) {
    arg->set_user_data<abstract::AbstractBase>(kUserDataAbs, nullptr);
  }
  out_value->set_user_data<abstract::AbstractBase>(kUserDataAbs, nullptr);
}

bool ApplySavedTensorsHooksOnK(const FuncGraphPtr &k, const FuncGraphPtr &current_primal_fg, const CNodePtr &cnode,
                               const FuncGraphManagerPtr &manager, const std::vector<AnfNodePtr> &transf_args) {
  MS_EXCEPTION_IF_NULL(k);
  MS_EXCEPTION_IF_NULL(manager);

  const auto &saved_tensors_hooks_stk = SavedTensorsHooks::Stack::GetInstance();
  FuncGraphPtr pack_hook = saved_tensors_hooks_stk.pack_hook();
  FuncGraphPtr unpack_hook = saved_tensors_hooks_stk.unpack_hook();
  if (pack_hook == nullptr && unpack_hook == nullptr) {
    return false;
  }

  MS_LOG(DEBUG) << "Applying saved tensors pack hook, pack_hook: " << (pack_hook ? pack_hook->ToString() : "null")
                << ", unpack_hook: " << (unpack_hook ? unpack_hook->ToString() : "null");

  auto pack_tensor = [&k, &pack_hook](const AnfNodePtr &tensor) {
    return (pack_hook == nullptr ? tensor : k->NewCNodeInOrder({NewValueNode(pack_hook), tensor}));
  };

  auto fprop_node = k->output();
  MS_EXCEPTION_IF_NULL(fprop_node);
  auto fprop_cnode = fprop_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(fprop_cnode);
  MS_EXCEPTION_IF_CHECK_FAIL(
    IsPrimitiveCNode(fprop_cnode, prim::kPrimMakeTuple) && fprop_cnode->size() == kKGraphOutPutCnodeSize,
    std::string("K's output must be a MakeTuple CNode with exactly two elements: (forward_output, bprop), but got ") +
      fprop_cnode->DebugString());
  auto bprop_env_fg = GetValueNode<FuncGraphPtr>(fprop_cnode->input(kCNodeSecondDataInput));
  MS_EXCEPTION_IF_NULL(bprop_env_fg);

  auto unpack_tensor = [&bprop_env_fg, &unpack_hook](const AnfNodePtr &packed_tensor) {
    return (unpack_hook == nullptr ? packed_tensor
                                   : bprop_env_fg->NewCNodeInFront({NewValueNode(unpack_hook), packed_tensor}));
  };

  auto out_value = fprop_cnode->input(kCNodeFirstDataInput);
  // Set Abstract to user_data temporarily.
  SetUserDataAbs(transf_args, out_value, cnode, current_primal_fg);

  auto useers_of_need_saved_tensors = UsersOfNeedSavedTensors(bprop_env_fg, k);
  if (useers_of_need_saved_tensors.empty()) {  // Some bprop may not use any saved tensors, eg: bprop of `add`.
    return false;
  }

  for (const auto &[saved_tensor, user_idxs] : useers_of_need_saved_tensors) {
    MS_LOG(DEBUG) << "Packing saved tensor: " + saved_tensor->ToString();
    auto packed_tensor = pack_tensor(saved_tensor);
    auto unpacked_tensor = unpack_tensor(packed_tensor);
    for (auto &[user, idx] : user_idxs) {
      MS_LOG(DEBUG) << "  Replacing input " << idx
                    << " of user " + user->ToString() + " with unpacked tensor " + unpacked_tensor->ToString() + ".";
      manager->SetEdge(user, idx, unpacked_tensor);
    }
  }

  // Clear user_data
  ClearUserDataAbs(transf_args, out_value);

  return true;
}
}  // namespace ad
}  // namespace mindspore
