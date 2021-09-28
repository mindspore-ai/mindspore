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

#include "backend/optimizer/ascend/format_type/rectify_do_mask_kernel_info.h"

#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "utils/utils.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef RectifyDoMaskKernelInfo::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({X, Xs});
}

const AnfNodePtr RectifyDoMaskKernelInfo::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    return RectifyKernelInfoInPynativeProcess(node);
  }
  if (AnfAlgo::GetCNodeName(cnode) != prim::kPrimDropoutGenMask->name()) {
    return nullptr;
  }
  std::vector<CNodePtr> do_mask_node_list;
  auto gen_mask_output_nodes = GetRealNodeUsedList(graph, cnode);
  MS_EXCEPTION_IF_NULL(gen_mask_output_nodes);
  for (const auto &output_node : *gen_mask_output_nodes) {
    if (AnfAlgo::GetCNodeName(output_node.first) == prim::kPrimDropoutDoMask->name()) {
      MS_EXCEPTION_IF_NULL(output_node.first);
      auto output_cnode = output_node.first->cast<CNodePtr>();
      do_mask_node_list.push_back(output_cnode);
    }
  }

  RectifyKernelInfo(do_mask_node_list, graph);
  return nullptr;
}

void RectifyDoMaskKernelInfo::RectifyKernelInfo(const std::vector<CNodePtr> &do_mask_node_list,
                                                const FuncGraphPtr &graph) const {
  std::map<std::string, size_t> format_counter;
  std::string special_format;
  std::string convert_format;
  for (const auto &do_mask : do_mask_node_list) {
    auto do_mask_data_format = AnfAlgo::GetInputFormat(do_mask, 0);
    if (special_format.empty() && kHWSpecialFormatSet.find(do_mask_data_format) != kHWSpecialFormatSet.end()) {
      special_format = do_mask_data_format;
    }
    if (format_counter.find(do_mask_data_format) == format_counter.end()) {
      format_counter[do_mask_data_format] = 1;
    } else {
      format_counter[do_mask_data_format] = format_counter[do_mask_data_format] + 1;
    }
  }
  if (format_counter.size() == 1) {
    return;
  }
  if (convert_format.empty()) {
    convert_format = GetConvertFormat(format_counter);
  }
  RectifyDropOutDoMaskKernelInfo(do_mask_node_list, convert_format, graph);
}

std::string RectifyDoMaskKernelInfo::GetConvertFormat(const std::map<std::string, size_t> &format_counter) const {
  constexpr size_t kFormatCount = 2;
  std::string convert_format = kOpFormat_DEFAULT;
  size_t counter = 0;
  if (format_counter.size() > kFormatCount) {
    return kOpFormat_DEFAULT;
  }
  if (format_counter.size() == kFormatCount && format_counter.find(kOpFormat_DEFAULT) == format_counter.end()) {
    return kOpFormat_DEFAULT;
  }
  for (const auto &iter : format_counter) {
    if (counter < iter.second) {
      convert_format = iter.first;
      counter = iter.second;
    } else if (counter == iter.second && kHWSpecialFormatSet.find(iter.first) != kHWSpecialFormatSet.end()) {
      convert_format = iter.first;
    }
  }
  return convert_format;
}

void RectifyDoMaskKernelInfo::RectifyDropOutDoMaskKernelInfo(const std::vector<CNodePtr> &do_mask_node_list,
                                                             const std::string &format,
                                                             const FuncGraphPtr &graph) const {
  for (const auto &do_mask : do_mask_node_list) {
    if (AnfAlgo::GetInputFormat(do_mask, 0) != format) {
      auto builder =
        std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(do_mask));
      MS_EXCEPTION_IF_NULL(builder);
      builder->SetInputFormat(format, 0);
      builder->SetOutputFormat(format, 0);
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), do_mask.get());
      ReSelecChildNodeKernelInfo(do_mask, graph);
    }
  }
}

AnfNodePtr RectifyDoMaskKernelInfo::RectifyKernelInfoInPynativeProcess(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return nullptr;
  }
  if (AnfAlgo::GetCNodeName(cnode) != prim::kPrimDropoutDoMask->name()) {
    return nullptr;
  }
  auto do_mask_input_format = AnfAlgo::GetInputFormat(node, 0);
  if (do_mask_input_format != kOpFormat_DEFAULT) {
    auto builder =
      std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
    MS_EXCEPTION_IF_NULL(builder);
    builder->SetInputFormat(kOpFormat_DEFAULT, 0);
    builder->SetOutputFormat(kOpFormat_DEFAULT, 0);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
  }
  return nullptr;
}

void RectifyDoMaskKernelInfo::ReSelecChildNodeKernelInfo(const CNodePtr &cnode, const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(cnode);
  auto output_node_list = GetRealNodeUsedList(graph, cnode);
  MS_EXCEPTION_IF_NULL(output_node_list);
  for (const auto &out_node_info : *output_node_list) {
    MS_EXCEPTION_IF_NULL(out_node_info.first);
    auto out_node = out_node_info.first->cast<CNodePtr>();
    if (AnfAlgo::IsRealKernel(out_node_info.first)) {
      auto ori_build_info = AnfAlgo::GetSelectKernelBuildInfo(out_node);
      kernel_selecter->SelectKernel(out_node);
      auto new_build_info = AnfAlgo::GetSelectKernelBuildInfo(out_node);
      MS_EXCEPTION_IF_NULL(new_build_info);
      MS_EXCEPTION_IF_NULL(ori_build_info);
      if ((*new_build_info) != (*ori_build_info)) {
        ReSelecChildNodeKernelInfo(out_node, graph);
      }
    } else if (AnfAlgo::GetCNodeName(out_node) == prim::kPrimTupleGetItem->name() ||
               AnfAlgo::GetCNodeName(out_node) == prim::kPrimDepend->name()) {
      ReSelecChildNodeKernelInfo(out_node, graph);
    } else {
      MS_LOG(INFO) << "Reselected the node " << cnode->DebugString() << " failed";
    }
  }
}
}  // namespace opt
}  // namespace mindspore
