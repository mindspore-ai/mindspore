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
#include "plugin/device/ascend/optimizer/format_type/convert_cast_format.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
namespace mindspore {
namespace opt {
const BaseRef ConvertCastFormat::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({X, Xs});
}

const AnfNodePtr ConvertCastFormat::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfUtils::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  auto node_name = common::AnfAlgo::GetCNodeName(node);
  if (node_name == prim::kPrimCast->name()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_node =
      common::AnfAlgo::VisitKernelWithReturnType(common::AnfAlgo::GetInputNode(cnode, input_index), 0).first;
    MS_EXCEPTION_IF_NULL(input_node);
    if (!input_node->isa<CNode>()) {
      continue;
    }
    auto cast_node = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cast_node);
    ChangeCastFormat(cast_node, func_graph);
  }
  return nullptr;
}

void ConvertCastFormat::SetCastFormat(const CNodePtr &cast_node, const string &format) const {
  MS_EXCEPTION_IF_NULL(cast_node);
  auto info_builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(cast_node));
  MS_EXCEPTION_IF_NULL(info_builder);
  info_builder->SetInputsFormat({format});
  info_builder->SetOutputsFormat({format});
  AnfAlgo::SetSelectKernelBuildInfo(info_builder->Build(), cast_node.get());
}

void ConvertCastFormat::ChangeCastFormat(const CNodePtr &cast_node, const FuncGraphPtr &func_graph) const {
  MS_EXCEPTION_IF_NULL(cast_node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto input_node_name = common::AnfAlgo::GetCNodeName(cast_node);
  if (input_node_name != prim::kPrimCast->name()) {
    return;
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrVisited, cast_node) &&
      common::AnfAlgo::GetNodeAttr<bool>(cast_node, kAttrVisited)) {
    return;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), cast_node);
  auto used_cast_node_list = GetRealNodeUsedList(func_graph, cast_node);
  MS_EXCEPTION_IF_NULL(used_cast_node_list);
  mindspore::HashMap<string, size_t> format_counter = CalculateFormat(used_cast_node_list, cast_node);
  auto cast_input_format = AnfAlgo::GetPrevNodeOutputFormat(cast_node, 0);
  string convert_format = kOpFormat_DEFAULT;
  if (cast_input_format == kOpFormat_DEFAULT) {
    SetCastFormat(cast_node, convert_format);
    return;
  }
  if (format_counter.size() == 1 && format_counter.begin()->first == kOpFormat_DEFAULT) {
    SetCastFormat(cast_node, convert_format);
    return;
  }
  auto it = format_counter.find(cast_input_format);
  if (it == format_counter.end()) {
    format_counter[cast_input_format] = 1;
  } else {
    it->second++;
  }
  if (format_counter.size() < 2) {
    size_t max_counter = 0;
    for (const auto &iter : format_counter) {
      if (iter.second > max_counter) {
        max_counter = iter.second;
        convert_format = iter.first;
      }
    }
    // change cast to default that can be more faster when it cast other hw format
    SetCastFormat(cast_node, convert_format);
  }
}

mindspore::HashMap<string, size_t> ConvertCastFormat::CalculateFormat(
  const std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> &used_cast_node_list,
  const CNodePtr &cast_node) const {
  MS_EXCEPTION_IF_NULL(used_cast_node_list);
  MS_EXCEPTION_IF_NULL(cast_node);
  mindspore::HashMap<string, size_t> format_counter;
  for (const auto &node_info : *used_cast_node_list) {
    MS_EXCEPTION_IF_NULL(node_info.first);
    auto cast_out_node = node_info.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cast_out_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(cast_out_node);
    for (size_t index = 0; index < input_num; ++index) {
      if (common::AnfAlgo::VisitKernelWithReturnType(
            common::AnfAlgo::GetInputNode(cast_out_node->cast<CNodePtr>(), index), 0)
            .first != cast_node) {
        continue;
      }
      auto format = AnfAlgo::GetInputFormat(cast_out_node, index);
      auto it = format_counter.find(format);
      if (it == format_counter.end()) {
        format_counter[format] = 1;
      } else {
        it->second++;
      }
    }
  }
  return format_counter;
}
}  // namespace opt
}  // namespace mindspore
