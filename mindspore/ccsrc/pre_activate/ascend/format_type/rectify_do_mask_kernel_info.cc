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

#include "pre_activate/ascend/format_type/rectify_do_mask_kernel_info.h"

#include <vector>
#include <map>
#include <string>
#include <memory>

#include "session/anf_runtime_algorithm.h"
#include "kernel/kernel_build_info.h"
#include "utils/utils.h"
#include "kernel/common_utils.h"
#include "utils/context/ms_context.h"

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
  if (ms_context->execution_mode() == kPynativeMode) {
    if (AnfAlgo::GetCNodeName(cnode) != prim::kPrimDropoutDoMask->name()) {
      return nullptr;
    }
    auto do_mask_input_format = AnfAlgo::GetInputFormat(node, 0);
    if (do_mask_input_format != kOpFormat_DEFAULT) {
      auto builder =
        std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
      builder->SetInputFormat(kOpFormat_DEFAULT, 0);
      builder->SetOutputFormat(kOpFormat_DEFAULT, 0);
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
    }
    return nullptr;
  }
  if (AnfAlgo::GetCNodeName(cnode) != prim::kPrimDropoutGenMask->name()) {
    return nullptr;
  }
  std::vector<CNodePtr> do_mask_node_list;
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_map = manager->node_users();
  auto iter = node_map.find(node);
  if (iter == node_map.end()) {
    MS_LOG(EXCEPTION) << "Cannot find the node " << node->DebugString() << " in the graph manager!";
  }
  auto gen_mask_output_nodes = iter->second;
  for (const auto &output_node : gen_mask_output_nodes) {
    if (AnfAlgo::GetCNodeName(output_node.first) == prim::kPrimDropoutDoMask->name()) {
      auto output_cnode = output_node.first->cast<CNodePtr>();
      do_mask_node_list.push_back(output_cnode);
    }
  }
  std::vector<size_t> input_shape;
  for (const auto &output_node : do_mask_node_list) {
    if (input_shape.empty()) {
      input_shape = AnfAlgo::GetPrevNodeOutputInferShape(output_node, 0);
      continue;
    }
    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(output_node, 0);
    if (!kernel::IsSameShape(shape, input_shape)) {
      MS_LOG(EXCEPTION) << "The DropOutGenMask connected with same genmask's shape must be equal!"
                        << " GenMask " << node->DebugString();
    }
  }
  RectifyKernelInfo(do_mask_node_list);
  return nullptr;
}

void RectifyDoMaskKernelInfo::RectifyKernelInfo(const std::vector<CNodePtr> &do_mask_node_list) const {
  std::map<std::string, size_t> format_counter;
  std::string special_format;
  std::string convert_format;
  for (const auto &do_mask : do_mask_node_list) {
    auto do_mask_data_format = AnfAlgo::GetInputFormat(do_mask, 0);
    if (special_format.empty() && kNeedTransFormatSet.find(do_mask_data_format) != kNeedTransFormatSet.end()) {
      special_format = do_mask_data_format;
    }
    if (format_counter.find(do_mask_data_format) == format_counter.end()) {
      format_counter[do_mask_data_format] = 1;
    } else {
      format_counter[do_mask_data_format] = format_counter[do_mask_data_format] + 1;
    }
    // if has two or more special format we need change all domask's format to default that can avoid insert more
    // transdata
    if (format_counter.size() > 2) {
      convert_format = kOpFormat_DEFAULT;
      break;
    }
    if (kNeedTransFormatSet.find(do_mask_data_format) != kNeedTransFormatSet.end() &&
        special_format != do_mask_data_format) {
      convert_format = kOpFormat_DEFAULT;
      break;
    }
  }
  if (format_counter.size() == 1) {
    return;
  }
  if (convert_format.empty()) {
    convert_format = GetConvertFormat(format_counter);
  }
  RectifyDropOutDoMaskKernelInfo(do_mask_node_list, convert_format);
}

std::string RectifyDoMaskKernelInfo::GetConvertFormat(const std::map<std::string, size_t> &format_counter) const {
  std::string convert_format;
  size_t counter = 0;
  for (const auto &iter : format_counter) {
    if (counter < iter.second) {
      convert_format = iter.first;
    }
    if (counter == iter.second && kNeedTransFormatSet.find(convert_format) == kNeedTransFormatSet.end()) {
      convert_format = iter.first;
    }
  }
  return convert_format;
}
void RectifyDoMaskKernelInfo::RectifyDropOutDoMaskKernelInfo(const std::vector<CNodePtr> &do_mask_node_list,
                                                             const std::string &format) const {
  for (const auto &do_mask : do_mask_node_list) {
    auto builder =
      std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(do_mask));
    builder->SetInputFormat(format, 0);
    builder->SetOutputFormat(format, 0);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), do_mask.get());
  }
}

}  // namespace opt
}  // namespace mindspore
