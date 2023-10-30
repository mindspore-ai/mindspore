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

#include "backend/common/pass/flatten_value_sequence_in_value_node.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/framework_ops.h"

namespace mindspore {
namespace opt {
namespace {
struct InputBuildInfo {
  std::vector<std::string> formats;
  std::vector<TypeId> device_types;
  std::vector<std::string> reshape_types;
  std::vector<kernel::KernelObjectType> kernel_object_types;
};

void FlattenSequence(const ValueSequencePtr &value_sequence, const FuncGraphPtr &func_graph,
                     std::vector<AnfNodePtr> *inputs, InputBuildInfo *input_build_info) {
  MS_EXCEPTION_IF_NULL(value_sequence);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(input_build_info);
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (const auto &value : value_sequence->value()) {
    MS_EXCEPTION_IF_NULL(value);
    if ((!value->isa<Scalar>()) && (!value->isa<tensor::Tensor>())) {
      MS_LOG(EXCEPTION) << "Invalid value:" << value->ToString();
    }
    const auto &value_node = NewValueNode(value);
    MS_EXCEPTION_IF_NULL(value_node);
    auto abstract = value->ToAbstract();
    value_node->set_abstract(abstract);
    auto kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    value_node->set_kernel_info(kernel_info);
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetOutputsFormat({kOpFormat_DEFAULT});
    MS_EXCEPTION_IF_NULL(value->type());
    builder.SetOutputsDeviceType({value->type()->type_id()});
    auto kernel_object_type =
      value->isa<Scalar>() ? kernel::KernelObjectType::SCALAR : kernel::KernelObjectType::TENSOR;
    builder.SetOutputsKernelObjectType({kernel_object_type});

    inputs->emplace_back(value_node);
    input_build_info->formats.emplace_back(kOpFormat_DEFAULT);
    input_build_info->device_types.emplace_back(value->type()->type_id());
    input_build_info->reshape_types.emplace_back("");
    input_build_info->kernel_object_types.emplace_back(kernel_object_type);

    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), value_node.get());
    kernel_graph->AddValueNodeToGraph(value_node);
  }
}
}  // namespace
const AnfNodePtr FlattenValueSequenceInValueNode::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                          const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPyExecute)) {
    return nullptr;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  const auto &input_formats = AnfAlgo::GetAllInputFormats(node);
  const auto &input_data_types = AnfAlgo::GetAllInputDeviceTypes(node);
  const auto &input_kernel_object_types = AnfAlgo::GetInputKernelObjectTypes(node);
  MS_LOG(DEBUG) << "check cnode:" << cnode->DebugString();
  if (inputs.size() != input_formats.size() + 1 || inputs.size() != input_data_types.size() + 1 ||
      inputs.size() != input_kernel_object_types.size() + 1) {
    MS_LOG(DEBUG) << "for node:" << node->DebugString() << " inputs size:" << inputs.size()
                  << " input format size:" << input_formats.size()
                  << " input device type size:" << input_data_types.size()
                  << " input kernel object type size:" << input_kernel_object_types.size();
    return nullptr;
  }
  auto input_reshape_types = AnfAlgo::GetAllInputReshapeType(node);
  if (inputs.size() != input_reshape_types.size() + 1) {
    MS_LOG(WARNING) << "Invalid input reshape type size:" << input_reshape_types.size()
                    << " and fix to:" << inputs.size() - 1;
    input_reshape_types = std::vector<std::string>(inputs.size() - 1, "");
    auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
    MS_EXCEPTION_IF_NULL(build_info);
    build_info->SetInputsReshapeType(input_reshape_types);
  }
  std::vector<AnfNodePtr> new_inputs;
  InputBuildInfo input_build_info;
  bool is_update = false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<ValueNode>()) {
      new_inputs.emplace_back(input);
      if (i != 0) {
        input_build_info.formats.emplace_back(input_formats[i - 1]);
        input_build_info.device_types.emplace_back(input_data_types[i - 1]);
        input_build_info.reshape_types.emplace_back(input_reshape_types[i - 1]);
        input_build_info.kernel_object_types.emplace_back(input_kernel_object_types[i - 1]);
      }
      continue;
    }
    const auto &value = input->cast<ValueNodePtr>()->value();
    if (value == nullptr || (!value->isa<ValueSequence>())) {
      new_inputs.emplace_back(input);
      if (i != 0) {
        input_build_info.formats.emplace_back(input_formats[i - 1]);
        input_build_info.device_types.emplace_back(input_data_types[i - 1]);
        input_build_info.reshape_types.emplace_back(input_reshape_types[i - 1]);
        input_build_info.kernel_object_types.emplace_back(input_kernel_object_types[i - 1]);
      }
      continue;
    }
    is_update = true;
    FlattenSequence(value->cast<ValueSequencePtr>(), func_graph, &new_inputs, &input_build_info);
  }
  if (is_update) {
    const auto &new_cnode = func_graph->NewCNode(new_inputs);
    MS_EXCEPTION_IF_NULL(new_cnode);
    MS_LOG(DEBUG) << "Update pyexecute node from:" << node->DebugString() << " to:" << new_cnode->DebugString()
                  << " input format size from:" << input_formats.size() << " to:" << input_build_info.formats.size();
    new_cnode->set_abstract(node->abstract());
    auto kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    new_cnode->set_kernel_info(kernel_info);
    auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
    MS_EXCEPTION_IF_NULL(build_info);
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(build_info);
    MS_EXCEPTION_IF_NULL(builder);
    builder->SetInputsFormat(input_build_info.formats);
    builder->SetInputsDeviceType(input_build_info.device_types);
    builder->SetInputsReshapeType(input_build_info.reshape_types);
    builder->SetInputsKernelObjectType(input_build_info.kernel_object_types);
    kernel_info->set_select_kernel_build_info(builder->Build());
    return new_cnode;
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
