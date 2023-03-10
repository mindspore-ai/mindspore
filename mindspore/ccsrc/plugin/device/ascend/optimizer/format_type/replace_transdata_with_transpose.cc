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

#include "plugin/device/ascend/optimizer/format_type/replace_transdata_with_transpose.h"
#include <string>
#include <memory>
#include <set>
#include <vector>
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
namespace {
const size_t kTransDataInputIndex = 1;

bool CheckTransDataSupport(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::IsDynamicShape(node)) {
    return true;
  }
  static std::set<string> format_list = {kOpFormat_DEFAULT, kOpFormat_NCHW, kOpFormat_CHWN, kOpFormat_HWCN,
                                         kOpFormat_NHWC};
  auto input_format = AnfAlgo::GetInputFormat(node, 0);
  auto output_format = AnfAlgo::GetOutputFormat(node, 0);
  auto input_shape = AnfAlgo::GetInputDeviceShape(node, 0);
  auto output_shape = AnfAlgo::GetOutputDeviceShape(node, 0);
  return format_list.find(input_format) == format_list.end() || format_list.find(output_format) == format_list.end() ||
         input_shape.size() != kDim4 || output_shape.size() != kDim4;
}

std::vector<int64_t> GetTransposePerm(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto input_format = AnfAlgo::GetInputFormat(node, 0);
  auto output_format = AnfAlgo::GetOutputFormat(node, 0);
  if (input_format == kOpFormat_DEFAULT) {
    input_format = kOpFormat_NCHW;
  }
  if (output_format == kOpFormat_DEFAULT) {
    output_format = kOpFormat_NCHW;
  }
  std::vector<int64_t> perm_value;
  for (size_t i = 0; i < output_format.size(); i++) {
    auto index = input_format.find((output_format[i]));
    if (index == std::string::npos) {
      MS_LOG(EXCEPTION) << "Can not find output dim [" << output_format[i] << "] in input format [" << input_format
                        << "].";
    }
    perm_value.emplace_back(index);
  }
  return perm_value;
}

ValueNodePtr CreatePermValueNode(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto perm = GetTransposePerm(node);
  auto perm_value = std::make_shared<tensor::Tensor>(perm, kInt64);
  auto perm_node = NewValueNode(perm_value);
  MS_EXCEPTION_IF_NULL(perm_node);
  auto value_abstract = perm_value->ToAbstract();
  perm_node->set_abstract(value_abstract);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  perm_node->set_kernel_info(kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat({kOpFormat_DEFAULT});
  builder.SetOutputsDeviceType({kNumberTypeInt64});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), perm_node.get());
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddValueNodeToGraph(perm_node);
  return perm_node;
}

void SetKernelBuildInfo(const AnfNodePtr &transdata, const AnfNodePtr &transpose) {
  MS_EXCEPTION_IF_NULL(transpose);
  MS_EXCEPTION_IF_NULL(transdata);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetInputsFormat({AnfAlgo::GetInputFormat(transdata, 0), kOpFormat_DEFAULT});
  selected_kernel_builder.SetInputsDeviceType(
    {AnfAlgo::GetInputDeviceDataType(transdata, 0), TypeId::kNumberTypeInt64});
  selected_kernel_builder.SetOutputsFormat({AnfAlgo::GetOutputFormat(transdata, 0)});
  selected_kernel_builder.SetOutputsDeviceType({AnfAlgo::GetOutputDeviceDataType(transdata, 0)});
  selected_kernel_builder.SetInputsKernelObjectType(
    {kernel::KernelObjectType::TENSOR, kernel::KernelObjectType::TENSOR});
  selected_kernel_builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  selected_kernel_builder.SetInputsReshapeType({"", ""});
  selected_kernel_builder.SetOutputsReshapeType({""});
  selected_kernel_builder.SetProcessor(kernel::Processor::AICORE);
  selected_kernel_builder.SetKernelType(KernelType::TBE_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), transpose.get());
}

AnfNodePtr CreateNewTranspose(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto perm_node = CreatePermValueNode(func_graph, node);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(kTransposeOpName)),
                                    cnode->input(kTransDataInputIndex), perm_node};
  auto transpose = pass.NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(transpose);
  SetKernelBuildInfo(node, transpose);
  std::vector<std::string> input_names = {"x", "perm"};
  std::vector<std::string> output_names = {"output"};
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), transpose);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), transpose);
  common::AnfAlgo::SetNodeAttr(kAttrForFormatChange, MakeValue(true), transpose);
  transpose->set_abstract(node->abstract());
  transpose->set_scope(node->scope());
  return transpose;
}
}  // namespace

const BaseRef ReplaceTransDataWithTranspose::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  return VectorRef({prim::kPrimTransData, x1});
}

const AnfNodePtr ReplaceTransDataWithTranspose::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Process node: " << node->fullname_with_scope();
  if (CheckTransDataSupport(node)) {
    MS_LOG(DEBUG) << "TransData is support, no need replace. node: " << node->fullname_with_scope();
    return nullptr;
  }
  auto transpose = CreateNewTranspose(func_graph, node, *this);
  MS_EXCEPTION_IF_NULL(transpose);
  MS_LOG(DEBUG) << "TransData is not supported from input format " << AnfAlgo::GetInputFormat(node, 0)
                << " to output format " << AnfAlgo::GetOutputFormat(node, 0)
                << " in dynamic shape scenario, replace TransData with Transpose."
                << " Origin TransData: " << node->fullname_with_scope()
                << ", New Transpose: " << transpose->fullname_with_scope();
  return transpose;
}
}  // namespace opt
}  // namespace mindspore
