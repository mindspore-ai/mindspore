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

#include "backend/optimizer/ascend/format_type/add_reformat_op.h"
#include <memory>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "base/core_ops.h"
#include "runtime/device/kernel_info.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
namespace {
AnfNodePtr InsertReFormatOp(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &in_node,
                            size_t idx) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(in_node);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> reformat_inputs;
  auto node_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  MS_EXCEPTION_IF_NULL(node_kernel_build_info);
  auto reformat_prim = std::make_shared<Primitive>(prim::kPrimReformat->name());
  reformat_inputs.push_back(NewValueNode(reformat_prim));
  reformat_inputs.push_back(in_node);
  auto reformat = func_graph->NewCNode(reformat_inputs);
  auto reformat_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  reformat_builder->SetInputsFormat({AnfAlgo::GetPrevNodeOutputFormat(node, idx)});
  reformat_builder->SetOutputsFormat({AnfAlgo::GetInputFormat(node, idx)});
  reformat_builder->SetInputsDeviceType({AnfAlgo::GetPrevNodeOutputDeviceDataType(node, idx)});
  reformat_builder->SetOutputsDeviceType({node_kernel_build_info->GetInputDeviceType(idx)});
  AnfAlgo::SetSelectKernelBuildInfo(reformat_builder->Build(), reformat.get());

  reformat->set_abstract(in_node->abstract());
  AnfAlgo::SetNodeAttr("nop_op", MakeValue(true), reformat);
  return reformat;
}

bool NeedInsert(const CNodePtr &cnode, const size_t input_index) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(cnode, input_index);
  auto real_input_node = kernel_with_index.first;
  auto idx = kernel_with_index.second;
  auto input_format = AnfAlgo::GetInputFormat(cnode, input_index);
  auto prev_format = AnfAlgo::GetOutputFormat(real_input_node, idx);
  bool flag_format = (input_format != prev_format);
  if (!flag_format) {
    return false;
  }

  bool flag_shape = true;
  auto input_origin_shape = AnfAlgo::GetOutputInferShape(real_input_node, idx);
  if (prev_format == kOpFormat_DEFAULT || input_format == kOpFormat_DEFAULT) {
    string checking_format = (prev_format == kOpFormat_DEFAULT) ? input_format : prev_format;
    // when input shape size is 1D, default format and NC1HWC0 are compatible
    if (input_origin_shape.size() == 1 && checking_format == kOpFormat_NC1HWC0) {
      flag_shape = false;
    }
    if (kDefaultCompatibleFormat.find(checking_format) != kDefaultCompatibleFormat.end()) {
      flag_shape = false;
    }
  }
  if (input_origin_shape.size() == 0) {
    flag_shape = false;
  }
  return flag_format && flag_shape;
}

AnfNodePtr NeedInSertReformatOp(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!node->isa<CNode>() || !AnfAlgo::IsRealKernel(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto in_nums = AnfAlgo::GetInputTensorNum(cnode);
  bool need_insert = false;
  std::vector<AnfNodePtr> new_inputs = {AnfAlgo::GetCNodePrimitiveNode(cnode)};
  for (size_t i = 0; i < in_nums; i++) {
    auto input_node = AnfAlgo::GetInputNode(cnode, i);
    if (NeedInsert(cnode, i)) {
      need_insert = true;
      auto re_format = InsertReFormatOp(func_graph, cnode, input_node, i);
      new_inputs.push_back(re_format);
      continue;
    }
    new_inputs.push_back(input_node);
  }
  if (need_insert) {
    auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
    CNodePtr new_node = nullptr;
    if (kernel_graph == nullptr) {
      new_node = std::make_shared<CNode>(*cnode);
    } else {
      new_node = kernel_graph->NewCNode(cnode);
    }
    MS_EXCEPTION_IF_NULL(new_node);
    new_node->set_inputs(new_inputs);
    AnfAlgo::CopyNodeAttrs(cnode, new_node);
    return new_node;
  }
  return nullptr;
}
}  // namespace

bool AddReFormatOp::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  bool changed = false;
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : node_list) {
    auto new_node = NeedInSertReformatOp(func_graph, node);
    if (new_node != nullptr) {
      manager->Replace(node, new_node);
      changed = true;
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
