/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/storage_format_convertor.h"

#include <queue>
#include <utility>
#include <vector>
#include <memory>
#include <string>
#include "graph/types.h"
#include "ops/conv_pool_ops.h"
#include "transform/graph_ir/storage_format_config_factory.h"
#include "ir/func_graph.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_info.h"
#include "transform/graph_ir/transform_util.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "ops/framework_op_name.h"
#include "ops/framework_ops.h"
#include "ops/nn_optimizer_ops.h"

namespace mindspore::transform {
namespace {
AnfNodePtr GetUsedOperator(const AnfNodePtr &node, const NodeUsersMap &node_users, const PrimitivePtr &prim) {
  auto iter = node_users.find(node);
  if (iter != node_users.end()) {
    for (const auto &node_user : iter->second) {
      if (common::AnfAlgo::GetCNodeName(node_user.first) == prim->name()) {
        return node_user.first;
      }
    }
  }
  return nullptr;
}

bool IsUsedByConv2D(const AnfNodePtr &node, const NodeUsersMap &node_users) {
  auto load = GetUsedOperator(node, node_users, prim::kPrimLoad);
  if (load == nullptr) {
    return false;
  }
  if (GetUsedOperator(load, node_users, prim::kPrimConv2D) != nullptr) {
    return true;
  }
  auto cast = GetUsedOperator(load, node_users, prim::kPrimCast);
  if (cast == nullptr) {
    return false;
  }
  return GetUsedOperator(cast, node_users, prim::kPrimConv2D) != nullptr;
}

bool IsUsedBySwitch(const AnfNodePtr &node, const NodeUsersMap &node_users) {
  if (common::AnfAlgo::GetCNodeName(node) != prim::kPrimPartial->name()) {
    return false;
  }

  if (GetUsedOperator(node, node_users, prim::kPrimSwitch) != nullptr) {
    return true;
  }

  auto make_tuple = GetUsedOperator(node, node_users, prim::kPrimMakeTuple);
  if (make_tuple != nullptr && GetUsedOperator(make_tuple, node_users, prim::kPrimSwitchLayer) != nullptr) {
    return true;
  }

  return false;
}

std::vector<std::pair<AnfNodePtr, int>> GetOutputNodesSkipVirtualNode(const FuncGraphManagerPtr &manager,
                                                                      const AnfNodePtr &node) {
  std::vector<std::pair<AnfNodePtr, int>> res;
  std::queue<std::pair<AnfNodePtr, int>> anf_queue;
  std::vector<AnfNodePtr> visited;
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users_map = manager->node_users();
  for (const auto &node_pair : node_users_map[node]) {
    anf_queue.push(node_pair);
    visited.push_back(node_pair.first);
  }
  while (!anf_queue.empty()) {
    auto queue_front = anf_queue.front();
    anf_queue.pop();
    // NOTE fix: do not support trans from NC1HWC0 to ND between parameter and Switch-op/switch_layer-op
    auto momentum_var = GetMomentumVarByAccum(node, node_users_map);
    if (IsUsedBySwitch(queue_front.first, node_users_map) && (momentum_var != nullptr) &&
        !IsUsedByConv2D(momentum_var, node_users_map)) {
      return {};
    }
    std::string op_name = common::AnfAlgo::GetCNodeName(queue_front.first);
    if (AnfUtils::IsRealKernel(queue_front.first) && op_name != kCastOpName && op_name != kTensorMoveOpName) {
      res.push_back(queue_front);
      continue;
    }
    for (const auto &node_pair : node_users_map[queue_front.first]) {
      if (std::find(visited.begin(), visited.end(), node_pair.first) != visited.end()) {
        continue;
      }
      anf_queue.push(node_pair);
      visited.push_back(node_pair.first);
    }
  }
  return res;
}
}  // namespace

AnfNodePtr GetMomentumVarByAccum(const AnfNodePtr &node, const NodeUsersMap &node_users) {
  auto param = node->cast<ParameterPtr>();
  if (param == nullptr) {
    return nullptr;
  }

  auto iter = node_users.find(node);
  if (iter == node_users.end()) {
    return nullptr;
  }

  for (const auto &param_user : iter->second) {
    auto cnode = param_user.first->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }

    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr || prim->name() != prim::kPrimApplyMomentum->name()) {
      continue;
    }

    auto accum = cnode->input(2)->cast<ParameterPtr>();
    if (accum == nullptr) {
      continue;
    }

    if (accum->name() == param->name()) {
      return cnode->input(1);
    }
  }

  return nullptr;
}

bool StorageFormatConvertor::SetupStorageFormat(const AnfGraphPtr &anf_graph, const AnfNodePtr &param,
                                                const std::shared_ptr<GeTensorDesc> &desc,
                                                const std::string &ori_format) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  MS_EXCEPTION_IF_NULL(param);
  MS_EXCEPTION_IF_NULL(desc);
  if (device::ascend::GetFormatMode() == "1" || !IsEnableRefMode()) {
    MS_LOG(DEBUG) << "Enable format mode or disable ref mode, no need to set storage format";
    return true;
  }

  auto param_ptr = param->cast<ParameterPtr>();
  if (param_ptr != nullptr && param_ptr->param_info() != nullptr &&
      !param_ptr->param_info()->storage_format().empty()) {
    std::string store_fmt = param_ptr->param_info()->storage_format();
    MS_LOG(INFO) << "Update desc format from set format: graph: " << anf_graph->ToString()
                 << ", storage format: " << store_fmt << ", pre param: " << param->DebugString()
                 << ", full name: " << param->ToString();
    auto format = GetGeFormat(param, store_fmt, desc->GetOriginShape().GetDimNum());
    UpdateTensorDesc(desc, format);
    UpdateParameterKernelInfo(param, store_fmt);
    return true;
  }

  std::string set_format;
  if (!InitParameterKernelInfo(param, &set_format)) {
    MS_LOG(INFO) << "Please attention: init Param kernel info failed.";
    return false;
  }
  if (set_format.empty()) {
    // The weight change storage format first time.
    SetStorageFormatFromConfig(anf_graph, param, desc);
  } else if (IsOneOfHWSpecialFormat(set_format)) {
    // The weight or data is from other subgraph or pynative node which has been set storage format.
    MS_LOG(INFO) << "Update desc format from set format: graph: " << anf_graph->ToString()
                 << ", storage format: " << set_format << ", pre param: " << param->DebugString()
                 << ", full name: " << param->ToString();
    auto format = GetGeFormat(param, set_format, desc->GetOriginShape().GetDimNum());
    UpdateTensorDesc(desc, format);
  }
  return true;
}

void StorageFormatConvertor::SetStorageFormatFromConfig(const AnfGraphPtr &anf_graph, const AnfNodePtr &param,
                                                        const std::shared_ptr<GeTensorDesc> &desc) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  MS_EXCEPTION_IF_NULL(param);
  MS_EXCEPTION_IF_NULL(desc);
  auto manager = anf_graph->manager();
  if (!manager) {
    MS_LOG(WARNING) << "Anf graph: " << anf_graph->ToString() << "'s manager is null. create a new one.";
    manager = Manage(anf_graph, true);
    anf_graph->set_manager(manager);
  }
  auto output_nodes = GetOutputNodesSkipVirtualNode(manager, param);
  for (const auto &user_node : output_nodes) {
    // Step 1: node storage format config
    auto op_type = common::AnfAlgo::GetCNodeName(user_node.first);
    auto storage_format_config_opt = StorageFormatConfigRegister::GetInstance().GetStorageFormatConfig(op_type);
    if (!storage_format_config_opt.has_value()) {
      continue;
    }
    auto &storage_format_config = storage_format_config_opt.value();
    // Step 2: node user index match
    auto storage_format_info_opt = storage_format_config.GetStorageFormatInfo(IntToSize(user_node.second));
    if (!storage_format_info_opt.has_value()) {
      continue;
    }
    // Step 3: check origin shape dims
    auto &storage_format_info = storage_format_info_opt.value();
    auto fmt_opt = storage_format_info.func_(user_node.first, desc);
    if (!fmt_opt.has_value()) {
      continue;
    }
    // Step 4: update desc and param format
    MS_EXCEPTION_IF_NULL(user_node.first);
    std::string store_fmt = fmt_opt.value();
    auto format = GetGeFormat(param, user_node.first, store_fmt, desc->GetOriginShape().GetDimNum());
    MS_LOG(INFO) << "Update desc format from config, graph: " << anf_graph->ToString()
                 << ", used node: " << user_node.first->DebugString() << ", full name: " << user_node.first->ToString()
                 << ",input idx: " << user_node.second << ", storage format: " << store_fmt
                 << ", pre param: " << param->DebugString() << ", full name: " << param->ToString();
    UpdateTensorDesc(desc, format);
    if (!storage_format_info.expand_dims_.empty()) {
      MS_LOG(INFO) << "Set expand dims rule stub.";
      // desc->SetExpandDimsRule(storage_format_info.expand_dims_);
    }
    UpdateParameterKernelInfo(param, store_fmt);
  }
}

void StorageFormatConvertor::UpdateTensorDesc(const std::shared_ptr<GeTensorDesc> &desc, int32_t format) {
  MS_EXCEPTION_IF_NULL(desc);
  desc->SetFormat(static_cast<ge::Format>(format));
  desc->SetShape({});
  desc->SetPlacement(ge::kPlacementDevice);
}

bool StorageFormatConvertor::InitParameterKernelInfo(const AnfNodePtr &param, std::string *format) {
  // param has default should have kernel info with one output
  MS_EXCEPTION_IF_NULL(param);
  MS_EXCEPTION_IF_NULL(format);
  const auto &output_with_indexes = common::AnfAlgo::GetAllOutputWithIndex(param);
  if (output_with_indexes.size() != 1) {
    MS_LOG(ERROR) << "Param: " << param->ToString() << "'s output size is not 1.";
    return false;
  }
  std::shared_ptr<device::KernelInfo> kernel_info =
    std::dynamic_pointer_cast<device::KernelInfo>(param->kernel_info_ptr());
  if (!kernel_info) {
    // create parameter node should create kernel info
    MS_LOG(INFO) << "Please attention, param: " << param->ToString() << "don't have kernel info.";
    return false;
  }
  kernel::KernelBuildInfoPtr build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  if (build_info && build_info->GetOutputDeviceType(0) != kTypeUnknown) {
    (*format) = build_info->GetOutputFormat(0);
    MS_LOG(INFO) << "Param: " << param->ToString() << " node has been setup, build info: " << build_info->ToString();
    return true;
  }

  if (!build_info) {
    MS_LOG(ERROR) << "Param: " << param->ToString() << " build info is null.";
    return false;
  }

  std::vector<TypeId> output_infer_types;
  std::vector<std::string> output_formats;
  (void)output_infer_types.emplace_back(common::AnfAlgo::GetOutputInferDataType(param, 0));
  (void)output_formats.emplace_back(kOpFormat_DEFAULT);
  build_info->SetOutputsDeviceType(output_infer_types);
  build_info->SetOutputsFormat(output_formats);
  kernel_info->set_select_kernel_build_info(build_info);
  return true;
}

void StorageFormatConvertor::UpdateParameterKernelInfo(const AnfNodePtr &param, const std::string &format) {
  // param has default should have kernel info with one output
  MS_EXCEPTION_IF_NULL(param);
  std::shared_ptr<device::KernelInfo> kernel_info =
    std::dynamic_pointer_cast<device::KernelInfo>(param->kernel_info_ptr());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel::KernelBuildInfoPtr build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(build_info);
  build_info->SetOutputsFormat({format});
  kernel_info->set_select_kernel_build_info(build_info);
}

int32_t StorageFormatConvertor::GetGeFormat(const AnfNodePtr &src_node, const AnfNodePtr &dst_node,
                                            const std::string &storage_format, size_t origin_dim) {
  MS_EXCEPTION_IF_NULL(src_node);
  MS_EXCEPTION_IF_NULL(dst_node);
  int64_t groups = 0;
  auto param = src_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param);
  auto cnode = dst_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kAttrGroups, cnode)) {
    groups = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrGroups);
    param->set_fracz_group(groups);
  }
  auto primary_format = TransformUtil::ConvertFormat(storage_format, origin_dim);
  auto format = ::ge::GetFormatFromSub(static_cast<int32_t>(primary_format), LongToInt(groups));
  return format;
}

int32_t StorageFormatConvertor::GetGeFormat(const AnfNodePtr &src_node, const std::string &storage_format,
                                            size_t origin_dim) {
  MS_EXCEPTION_IF_NULL(src_node);
  auto param = src_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param);
  auto primary_format = TransformUtil::ConvertFormat(storage_format, origin_dim);
  auto format = ::ge::GetFormatFromSub(static_cast<int32_t>(primary_format), LongToInt(param->fracz_group()));
  return format;
}
}  // namespace mindspore::transform
