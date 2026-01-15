/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#include "utils/hyper_offload/hyper_offload_utils.h"

#include <algorithm>
#include <optional>

#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "utils/compile_config.h"
#include "utils/anf_utils.h"
#include "include/utils/utils.h"
#include "include/utils/tensor_py.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "abstract/abstract_value.h"
#include "utils/hyper_offload/hyper_offload_strategy.h"
#include "utils/hyper_offload/hyper_offload_operation.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {
std::string GetParameterDeviceStr(const AnfNodePtr &node) {
  constexpr auto kParameterDeviceUserDataName = "parameter_device";
  if (!node->isa<Parameter>()) {
    return "";
  }
  const auto &parameter = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(parameter);
  const auto value = parameter->default_param();
  if (value == nullptr) {
    return "";
  }
  const auto meta_tensor = value->cast_ptr<tensor::MetaTensor>();
  if (meta_tensor == nullptr) {
    return "";
  }
  const auto &user_data = meta_tensor->user_data<tensor::TensorPyUserData>(kParameterDeviceUserDataName);
  if (user_data == nullptr || !py::isinstance<py::str>(user_data->obj)) {
    return "";
  }
  return py::cast<std::string>(user_data->obj);
}
}  // namespace
HyperOffloadConfig &HyperOffloadConfig::GetInstance() {
  static HyperOffloadConfig instance;
  return instance;
}

HyperOffloadConfig::HyperOffloadConfig() {
  enable_hyper_offload_ = (common::GetEnv("MS_DEV_HYPER_OFFLOAD") == "1");
  if (enable_hyper_offload_) {
    select_distance_ = std::stoul(common::GetCompileConfig("HYPER_OFFLOAD_SELECT_DISTANCE"));
    select_num_ = std::stoul(common::GetCompileConfig("HYPER_OFFLOAD_SELECT_NUM"));
    prefetch_distance_ = std::stoul(common::GetCompileConfig("HYPER_OFFLOAD_PREFETCH_DISTANCE"));
    release_distance_ = std::stoul(common::GetCompileConfig("HYPER_OFFLOAD_RELEASE_DISTANCE"));
  }
}

bool IsD2HNode(const AnfNodePtr &node) {
  return IsOneOfPrimitiveCNode(node, {prim::kPrimCopyToHost, prim::kPrimCopyToHostExt});
}

bool IsH2DNode(const AnfNodePtr &node) { return IsOneOfPrimitiveCNode(node, {prim::kPrimCopyToDevice}); }

bool IsViewOrInplaceNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  const auto &prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
  return prim != nullptr && (prim->graph_view_prim() || prim->inplace_prim());
}

bool IsValidOffloadNode(const UserInfo &user_info) {
  auto node = user_info.node.first;
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    return false;
  }
  if (IsOneOfPrimitiveCNode(node, {prim::kPrimMoveTo, prim::kPrimTensorMove, prim::kPrimConditionSwitch})) {
    return false;
  }
  if (std::any_of(user_info.users.begin(), user_info.users.end(), [](const auto &e) { return IsD2HNode(e.first); })) {
    return false;
  }
  return true;
}

UserInfoList CollectAllNodeUsers(const CNodePtrList &nodes) {
  UserInfoList user_info_list;
  for (const CNodePtr &node : nodes) {
    for (size_t i = 0; i < node->inputs().size(); ++i) {
      const auto &input = node->input(i);
      if (input->abstract() == nullptr || !input->abstract()->isa<abstract::AbstractTensor>()) {
        continue;
      }
      const auto &cur_kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(input, 0, false);
      if (!cur_kernel_with_index.first->isa<CNode>()) {
        continue;
      }
      auto iter = std::find_if(user_info_list.begin(), user_info_list.end(),
                               [&cur_kernel_with_index](const auto &e) { return cur_kernel_with_index == e.node; });
      auto cur_user = KernelWithIndex{node, i};
      if (iter == user_info_list.end()) {
        auto new_user_info = UserInfo{cur_kernel_with_index, {cur_user}};
        (void)user_info_list.emplace_back(new_user_info);
      } else {
        (void)(iter->users).emplace_back(cur_user);
      }
    }
  }
  return user_info_list;
}

std::optional<AnfNodePtr> FindLastRealUser(const AnfNodePtr &node, const UserInfoList &user_info_list) {
  int64_t output_index = 0;
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    auto cnode = node->cast<CNodePtr>();
    constexpr size_t index_pos = 2;
    auto index_value_node = cnode->input(index_pos)->cast<ValueNodePtr>();
    output_index = GetValue<int64_t>(index_value_node->value());
  }
  auto node_iter = std::find_if(user_info_list.begin(), user_info_list.end(), [&node, output_index](const auto &e) {
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      auto real_data = node->cast<CNodePtr>()->input(1);
      return e.node.first == real_data && (int64_t)e.node.second == output_index;
    }
    return e.node.first == node && (int64_t)e.node.second == output_index;
  });
  if (node_iter == user_info_list.end()) {
    return std::nullopt;
  }
  const auto &node_users = node_iter->users;
  auto last_user = node_users[node_users.size() - 1].first;
  // ToRemote will be placed right after the data is changed.
  if (IsD2HNode(last_user)) {
    return std::nullopt;
  }
  return last_user;
}

CNodePtr BuildToHostNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node,
                         const HandleNewNodeCallback &callback) {
  MS_EXCEPTION_IF_NULL(graph);
  const std::string &prim_name = prim::kPrimCopyToHost->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  AnfNodePtrList inputs{NewValueNode(prim), data_node};
  auto sync_node = NewValueNode(MakeValue(false));
  sync_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  (void)inputs.emplace_back(sync_node);
  auto ret_node = graph->NewCNode(inputs);
  ret_node->set_abstract(data_node->abstract());
  if (callback != nullptr) {
    callback(ret_node, data_node);
  }
  return ret_node;
}

CNodePtr BuildInplaceToHostNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node, const AnfNodePtr &update_node,
                                const AnfNodePtr &depend_node, const HandleNewNodeCallback &callback) {
  MS_EXCEPTION_IF_NULL(graph);
  const std::string &prim_name = prim::kPrimCopyToHostExt->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  AnfNodePtrList inputs{NewValueNode(prim), data_node, update_node, depend_node};
  auto sync_node = NewValueNode(MakeValue(false));
  sync_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  (void)inputs.emplace_back(sync_node);
  auto ret_node = graph->NewCNode(inputs);
  ret_node->set_abstract(data_node->abstract());
  if (callback != nullptr) {
    callback(ret_node, depend_node);
  }
  return ret_node;
}

CNodePtr BuildToDeviceNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node,
                           const HandleNewNodeCallback &callback) {
  MS_EXCEPTION_IF_NULL(graph);
  const std::string &prim_name = prim::kPrimCopyToDevice->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  auto depend_node = NewValueNode(MakeValue(false));
  depend_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  auto ret_node = graph->NewCNode({NewValueNode(prim), data_node, depend_node, depend_node});
  ret_node->set_abstract(data_node->abstract());
  if (callback != nullptr) {
    callback(ret_node, data_node);
  }
  return ret_node;
}

std::optional<ParameterPtr> FindParameterByRefName(const FuncGraphPtr &graph, const std::string &name) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &input : graph->parameters()) {
    if (!input->isa<Parameter>()) {
      continue;
    }
    auto param = input->cast<ParameterPtr>();
    if (param->name() == name) {
      return param;
    }
  }
  return std::nullopt;
}

std::optional<ParameterPtr> GetRemoteParameter(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abstract = node->abstract();
  if (abstract == nullptr) {
    return std::nullopt;
  }
  if (!abstract->isa<abstract::AbstractRefTensor>()) {
    return std::nullopt;
  }
  auto abstract_ref = abstract->cast<abstract::AbstractRefPtr>();
  const auto &ref_value = abstract_ref->ref_key_value()->cast<StringImmPtr>();
  MS_EXCEPTION_IF_NULL(ref_value);
  const auto &ref_name = ref_value->value();
  const auto &ref_node_res = FindParameterByRefName(graph, ref_name);
  if (!ref_node_res.has_value()) {
    return std::nullopt;
  }
  const auto &parameter = ref_node_res.value();
  const auto &device_str = GetParameterDeviceStr(parameter);
  if (device_str != kToRemote) {
    return std::nullopt;
  }
  return parameter;
}

void UpdateNodeInput(const CNodePtr &node, const size_t input_index, const AnfNodePtr &new_input_node) {
  MS_EXCEPTION_IF_NULL(node);
  node->set_input(input_index, new_input_node);
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
