/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/format_type/rectify_do_mask_kernel_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
// used to store the cross subgraph domask format
std::map<std::string, std::string> rectify_do_mask_map;

void RectifyKernelInfoInPynativeProcess(const CNodePtr &cnode) {
  auto cnode_name = common::AnfAlgo::GetCNodeName(cnode);
  if (cnode_name != prim::kPrimDropOutDoMask->name() && cnode_name != prim::kPrimDropOutDoMaskV3->name()) {
    return;
  }
  auto do_mask_input_format = AnfAlgo::GetInputFormat(cnode, 0);
  if (do_mask_input_format != kOpFormat_DEFAULT) {
    auto builder =
      std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(cnode));
    MS_EXCEPTION_IF_NULL(builder);
    builder->SetInputFormat(kOpFormat_DEFAULT, 0);
    builder->SetOutputFormat(kOpFormat_DEFAULT, 0);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), cnode.get());
  }
}

bool IsDoMask(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    auto node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (IsPrimitive(node, prim::kPrimDropOutDoMask) || IsPrimitive(node, prim::kPrimDropOutDoMaskV3)) {
      return true;
    }
  }
  return false;
}
}  // namespace

const BaseRef RectifyDoMaskKernelInfo::DefinePattern() const {
  VarPtr X = std::make_shared<CondVar>(IsDoMask);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({X, Xs});
}

void RectifyDoMaskKernelInfo::ReSelectChildNodeKernelInfo(const CNodePtr &cnode, const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(cnode);
  auto output_node_list = GetRealNodeUsedList(graph, cnode);
  MS_EXCEPTION_IF_NULL(output_node_list);
  for (const auto &out_node_info : *output_node_list) {
    MS_EXCEPTION_IF_NULL(out_node_info.first);
    auto out_node = out_node_info.first->cast<CNodePtr>();
    if (AnfUtils::IsRealKernel(out_node_info.first)) {
      auto ori_build_info = AnfAlgo::GetSelectKernelBuildInfo(out_node);
      kernel_selecter->SelectKernel(out_node);
      auto new_build_info = AnfAlgo::GetSelectKernelBuildInfo(out_node);
      MS_EXCEPTION_IF_NULL(new_build_info);
      MS_EXCEPTION_IF_NULL(ori_build_info);
      if ((*new_build_info) != (*ori_build_info)) {
        ReSelectChildNodeKernelInfo(out_node, graph);
      }
    } else if (common::AnfAlgo::GetCNodeName(out_node) == prim::kPrimTupleGetItem->name() ||
               common::AnfAlgo::GetCNodeName(out_node) == prim::kPrimDepend->name()) {
      ReSelectChildNodeKernelInfo(out_node, graph);
    } else {
      MS_LOG(INFO) << "Reselected the node " << cnode->DebugString() << " failed";
    }
  }
}

const AnfNodePtr RectifyDoMaskKernelInfo::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  auto do_mask_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(do_mask_node);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    RectifyKernelInfoInPynativeProcess(do_mask_node);
    return nullptr;
  }
  if (!do_mask_node->HasPrimalAttr(kPrimalAttrUniqueId) && !do_mask_node->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
    MS_LOG(INFO) << "The DoMask cnode has no primal attr: " << do_mask_node->DebugString();
    return nullptr;
  }
  std::string unique_id;
  if (do_mask_node->HasPrimalAttr(kPrimalAttrUniqueId)) {
    unique_id = do_mask_node->GetPrimalAttr(kPrimalAttrUniqueId)->DumpText();
  } else if (do_mask_node->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
    unique_id = do_mask_node->GetPrimalAttr(kPrimalAttrForwardUniqueId)->DumpText();
  }
  auto do_mask_data_format = AnfAlgo::GetInputFormat(do_mask_node, 0);
  auto iter = rectify_do_mask_map.find(unique_id);
  if (iter == rectify_do_mask_map.end()) {
    rectify_do_mask_map.insert({unique_id, do_mask_data_format});
  } else {
    auto &format = iter->second;
    MS_LOG(INFO) << "Node: " << do_mask_node->DebugString() << " need convert origin format: " << do_mask_data_format
                 << " to new format: " << format;
    if (format != do_mask_data_format) {
      auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(
        AnfAlgo::GetSelectKernelBuildInfo(do_mask_node));
      MS_EXCEPTION_IF_NULL(builder);
      builder->SetInputFormat(format, 0);
      builder->SetOutputFormat(format, 0);
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), do_mask_node.get());
      ReSelectChildNodeKernelInfo(do_mask_node, do_mask_node->func_graph());
    }
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
