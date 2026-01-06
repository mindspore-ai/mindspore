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

#include "plugin/ascend/kernel_executor/hyper_offload/hyper_offload_utils.h"

#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "plugin/ascend/kernel_executor/kernel_select_ascend.h"
#include "plugin/ascend/kernel_executor/rts/rt_kernel_build.h"
#include "plugin/ascend/graph_optimizer/stream_assign/acl_stream_assign.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace hyper_offload {
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

CNodePtr BuildToHostNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &data_node) {
  const std::string &prim_name = prim::kPrimCopyToHost->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  AnfNodePtrList inputs{NewValueNode(prim), data_node};
  auto sync_node = kernel_graph->NewValueNode(MakeValue(false));
  sync_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  (void)inputs.emplace_back(sync_node);
  auto ret_node = kernel_graph->NewCNode(inputs);
  ret_node->set_abstract(data_node->abstract());
  device::ascend::GenerateKernelBuildInfo(ret_node, RT_KERNEL);
  auto kernel_mod_ptr = kernel::RtOpBuild(ret_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  AnfAlgo::SetKernelMod(kernel_mod_ptr, ret_node.get());
  return ret_node;
}

CNodePtr BuildInplaceToHostNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &data_node,
                                const AnfNodePtr &update_node, const AnfNodePtr &depend_node) {
  const std::string &prim_name = prim::kPrimCopyToHostExt->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  AnfNodePtrList inputs{NewValueNode(prim), data_node, update_node, depend_node};
  auto sync_node = kernel_graph->NewValueNode(MakeValue(false));
  sync_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  (void)inputs.emplace_back(sync_node);
  auto ret_node = kernel_graph->NewCNode(inputs);
  ret_node->set_abstract(data_node->abstract());
  device::ascend::GenerateKernelBuildInfo(ret_node, RT_KERNEL);
  auto kernel_mod_ptr = kernel::RtOpBuild(ret_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  AnfAlgo::SetKernelMod(kernel_mod_ptr, ret_node.get());
  return ret_node;
}

CNodePtr BuildToDeviceNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &data_node) {
  const std::string &prim_name = prim::kPrimCopyToDevice->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  auto depend_node = kernel_graph->NewValueNode(MakeValue(false));
  depend_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  auto ret_node = kernel_graph->NewCNode({NewValueNode(prim), data_node, depend_node, depend_node});
  ret_node->set_abstract(data_node->abstract());
  device::ascend::GenerateKernelBuildInfo(ret_node, RT_KERNEL);
  auto kernel_mod_ptr = kernel::RtOpBuild(ret_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  AnfAlgo::SetKernelMod(kernel_mod_ptr, ret_node.get());
  return ret_node;
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

std::optional<ParameterPtr> FindParameterByRefName(const KernelGraphPtr &kernel_graph, const std::string &name) {
  for (const auto &input : kernel_graph->parameters()) {
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

std::optional<ParameterPtr> GetRemoteParameter(const KernelGraphPtr &kernel_graph, const mindspore::AnfNodePtr &node) {
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
  const auto &ref_node_res = FindParameterByRefName(kernel_graph, ref_name);
  if (!ref_node_res.has_value()) {
    return std::nullopt;
  }
  const auto &parameter = ref_node_res.value();
  const auto &device_str = AnfAlgo::GetParameterDeviceStr(parameter);
  if (device_str != kToRemote) {
    return std::nullopt;
  }
  return parameter;
}

void UpdateNodeInput(const KernelGraphPtr &kernel_graph, const CNodePtr &node, const size_t input_index,
                     const AnfNodePtr &new_input_node) {
  node->set_input(input_index, new_input_node);
  kernel_graph->ReplaceRefMapInput({node, 0}, {new_input_node, 0});
}
}  // namespace hyper_offload
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
