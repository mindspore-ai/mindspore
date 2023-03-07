/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/graph_kernel/converter/akg/kernel_builder.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "tools/graph_kernel/converter/akg/akg_build.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ops/primitive_c.h"
#include "ops/custom.h"
#include "utils/anf_utils.h"
#include "utils/log_adapter.h"

namespace mindspore::graphkernel {
namespace {
void BuildAKGKernel(const std::vector<AnfNodePtr> &node_list) {
  AnfNodePtrList anf_list;
  for (auto &node : node_list) {
    if (AnfUtils::IsGraphKernel(node)) {
      anf_list.push_back(node);
    }
  }
  if (anf_list.empty()) {
    return;
  }
  graphkernel::AkgKernelBuilder gk;
  if (!gk.CompileJsonsInAnfnodes(anf_list)) {
    MS_LOG(EXCEPTION) << "Graph kernel compile fail";
  }
}

std::string GetCNodeDynamicInputIndex(const CNodePtr &cnode) {
  std::string dynamic_input_index;
  auto cb = Callback::Instance();
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    if (cnode->input(i)->isa<CNode>() || cnode->input(i)->isa<Parameter>()) {
      auto input_shape = cb->GetInputShape(cnode, i - 1);
      if (input_shape.size() <= 0 || input_shape[0] != 1) {
        MS_LOG(EXCEPTION) << "Dynamic inputs' batch size should be 1";
      }
      dynamic_input_index += std::to_string(i - 1) + ",";
    }
  }
  return dynamic_input_index;
}

std::string GetCNodeInputShapeStr(const CNodePtr &cnode) {
  std::string input_shape_str;
  auto cb = Callback::Instance();
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_shape = cb->GetInputShape(cnode, i - 1);
    input_shape_str += std::to_string(input_shape.size()) + ",";
    for (auto &v : input_shape) {
      input_shape_str += std::to_string(v) + ",";
    }
  }
  return input_shape_str;
}

std::string GetCNodeOutputShapeStr(const CNodePtr &cnode) {
  std::string output_shape_str;
  auto output_num = AnfUtils::GetOutputTensorNum(cnode);
  auto cb = Callback::Instance();
  for (size_t i = 0; i < output_num; i++) {
    auto output_shape = cb->GetOutputShape(cnode, i);
    output_shape_str += std::to_string(output_shape.size()) + ",";
    for (auto &v : output_shape) {
      output_shape_str += std::to_string(v) + ",";
    }
  }
  return output_shape_str;
}

std::string GetCNodeOutputTypeStr(const CNodePtr &cnode) {
  std::string output_type_str;
  auto output_num = AnfUtils::GetOutputTensorNum(cnode);
  auto cb = Callback::Instance();
  for (size_t i = 0; i < output_num; i++) {
    auto output_type = cb->GetOutputType(cnode, i);
    output_type_str += std::to_string(static_cast<int>(output_type)) + ",";
  }
  return output_type_str;
}

std::string GetCNodeOutputFormatStr(const CNodePtr &cnode) {
  std::string output_format_str;
  auto output_num = AnfUtils::GetOutputTensorNum(cnode);
  auto cb = Callback::Instance();
  for (size_t i = 0; i < output_num; i++) {
    auto output_format = cb->GetOutputFormat(cnode, i);
    if (output_format == kOpFormat_NHWC) {
      output_format_str += "1,";
    } else {  // default, NCHW
      output_format_str += "0,";
    }
  }
  return output_format_str;
}
}  // namespace

AnfNodePtr KernelBuilder::CreateCustomOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  if (func_graph == nullptr || cnode == nullptr) {
    return nullptr;
  }
  auto op = std::make_shared<ops::Custom>();
  op->set_type("GraphKernel");
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  auto fg = GetCNodeFuncGraph(cnode);
  MS_EXCEPTION_IF_NULL(fg);
  auto kernel_name = GetValue<std::string>(fg->get_attr("kernel_name"));
  std::vector<uint8_t> kernel_name_str(kernel_name.begin(), kernel_name.end());
  custom_attrs["kernel_name"] = kernel_name_str;
  if (GraphKernelFlags::GetInstance().enable_dynamic_batch && fg->has_attr("dynamic_input_index")) {
    std::string dynamic_input_index = GetValue<std::string>(fg->get_attr("dynamic_input_index"));
    custom_attrs["dynamic_input_index"] = std::vector<uint8_t>(dynamic_input_index.begin(), dynamic_input_index.end());
  }
  std::string input_shape_str = GetCNodeInputShapeStr(cnode);
  std::string output_shape_str = GetCNodeOutputShapeStr(cnode);
  std::string output_format_str = GetCNodeOutputFormatStr(cnode);
  std::string output_type_str = GetCNodeOutputTypeStr(cnode);
  custom_attrs["inputs_shape"] = std::vector<uint8_t>(input_shape_str.begin(), input_shape_str.end());
  custom_attrs["outputs_shape"] = std::vector<uint8_t>(output_shape_str.begin(), output_shape_str.end());
  custom_attrs["outputs_format"] = std::vector<uint8_t>(output_format_str.begin(), output_format_str.end());
  custom_attrs["outputs_type"] = std::vector<uint8_t>(output_type_str.begin(), output_type_str.end());
  op->set_attr(custom_attrs);
  auto inputs = cnode->inputs();
  inputs[0] = NewValueNode(op->GetPrim());
  auto custom_cnode = func_graph->NewCNode(inputs);
  custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
  custom_cnode->set_abstract(cnode->abstract()->Clone());
  return custom_cnode;
}

bool KernelBuilder::Run(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  bool changed = false;
  if (GraphKernelFlags::GetInstance().enable_dynamic_batch) {
    for (auto &node : node_list) {
      if (!AnfUtils::IsGraphKernel(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      auto gk_fg = GetCNodeFuncGraph(cnode);
      MS_EXCEPTION_IF_NULL(gk_fg);
      std::string dynamic_input_index = GetCNodeDynamicInputIndex(cnode);
      if (!dynamic_input_index.empty()) {
        gk_fg->set_attr("dynamic_input_index", MakeValue(dynamic_input_index));
        changed = true;
      }
    }
  }

  BuildAKGKernel(node_list);
  auto manager = Manage(func_graph, true);
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : node_list) {
    if (!AnfUtils::IsGraphKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto custom_cnode = CreateCustomOp(func_graph, cnode);
    if (custom_cnode == nullptr) {
      MS_LOG(EXCEPTION) << "Create custom op fail for " << cnode->fullname_with_scope();
    }
    manager->Replace(node, custom_cnode);
    changed = true;
  }
  return changed;
}
}  // namespace mindspore::graphkernel
