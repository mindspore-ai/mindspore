/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/adapter/callback_impl.h"

#include <algorithm>
#include <vector>
#include <utility>
#include <memory>
#include "utils/ms_context.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/adapter/fake_abstract_shape.h"
#include "kernel/graph_kernel_info.h"

namespace mindspore::graphkernel {
GRAPH_KERNEL_CALLBACK_REGISTER(CallbackImpl);
ShapeVector CallbackImpl::GetInputShape(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetInputDeviceShape(node, i);
}

ShapeVector CallbackImpl::GetOutputShape(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetOutputDeviceShape(node, i);
}

ShapeVector CallbackImpl::GetInputInferShape(const AnfNodePtr &node, size_t i) {
  return common::AnfAlgo::GetPrevNodeOutputInferShape(node, i);
}

ShapeVector CallbackImpl::GetOutputInferShape(const AnfNodePtr &node, size_t i) {
  return common::AnfAlgo::GetOutputInferShape(node, i);
}

TypeId CallbackImpl::GetInputType(const AnfNodePtr &node, size_t i) { return AnfAlgo::GetInputDeviceDataType(node, i); }

TypeId CallbackImpl::GetOutputType(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetOutputDeviceDataType(node, i);
}

TypeId CallbackImpl::GetInputInferType(const AnfNodePtr &node, size_t i) {
  return common::AnfAlgo::GetPrevNodeOutputInferDataType(node, i);
}

TypeId CallbackImpl::GetOutputInferType(const AnfNodePtr &node, size_t i) {
  return common::AnfAlgo::GetOutputInferDataType(node, i);
}

std::string CallbackImpl::GetInputFormat(const AnfNodePtr &node, size_t i) { return AnfAlgo::GetInputFormat(node, i); }

std::string CallbackImpl::GetOutputFormat(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetOutputFormat(node, i);
}

std::string CallbackImpl::GetProcessor(const AnfNodePtr &node) { return kernel::GetProcessorStr(node); }

std::string CallbackImpl::GetTargetFromContext() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
}

void CallbackImpl::CollectInputTypesAndFormats(const AnfNodePtr &node, std::vector<TypeId> *input_types,
                                               std::vector<std::string> *input_formats, bool is_basic_node) {
  auto kernel_with_index = AnfUtils::VisitKernel(node, 0);
  if (kernel_with_index.first->isa<ValueNode>()) {
    auto tensor = GetValueNode<tensor::TensorPtr>(kernel_with_index.first);
    MS_EXCEPTION_IF_NULL(tensor);
    (void)input_formats->emplace_back(kOpFormat_DEFAULT);
    (void)input_types->emplace_back(tensor->data_type());
  } else if (kernel_with_index.first->isa<Parameter>()) {
    if (is_basic_node == false) {
      (void)input_formats->emplace_back(kOpFormat_DEFAULT);
      auto input_type = GetOutputInferType(kernel_with_index.first, kernel_with_index.second);
      (void)input_types->emplace_back(input_type);
    } else {
      auto input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
      (void)input_formats->emplace_back(std::move(input_format));
      auto input_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
      (void)input_types->emplace_back(input_type);
    }
  } else {
    auto input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    (void)input_formats->emplace_back(std::move(input_format));
    auto input_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
    (void)input_types->emplace_back(input_type);
  }
}

void CallbackImpl::SetGraphKernelNodeKernelInfo(const AnfNodePtr &node) {
  std::vector<std::string> graph_input_format;
  std::vector<TypeId> graph_input_type;
  std::vector<std::string> graph_output_format;
  std::vector<TypeId> graph_output_type;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = GetCNodeFuncGraph(node);
  MS_EXCEPTION_IF_NULL(fg);
  auto &inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    CollectInputTypesAndFormats(inputs[i], &graph_input_type, &graph_input_format);
    fg->parameters()[i - 1]->set_kernel_info(std::make_shared<device::KernelInfo>());
    kernel::KernelBuildInfo::KernelBuildInfoBuilder para_info_builder;
    para_info_builder.SetOutputsFormat({graph_input_format.back()});
    para_info_builder.SetOutputsDeviceType({graph_input_type.back()});
    para_info_builder.SetKernelType(KernelType::AKG_KERNEL);
    para_info_builder.SetProcessor(kernel::GetProcessorFromContext());
    AnfAlgo::SetSelectKernelBuildInfo(para_info_builder.Build(), fg->parameters()[i - 1].get());
  }
  AnfNodePtrList outputs;
  if (IsPrimitiveCNode(fg->output(), prim::kPrimMakeTuple)) {
    auto fg_output = fg->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(fg_output);
    outputs.assign(fg_output->inputs().begin() + 1, fg_output->inputs().end());
  } else {
    outputs.push_back(fg->output());
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto kernel_with_index = common::AnfAlgo::VisitKernel(outputs[i], 0);
    graph_output_format.push_back(AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second));
    graph_output_type.push_back(AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second));
  }
  kernel::KernelBuildInfo::KernelBuildInfoBuilder graph_info_builder;
  graph_info_builder.SetProcessor(kernel::GetProcessorFromContext());
  graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  graph_info_builder.SetFusionType(kernel::kPatternOpaque);
  graph_info_builder.SetInputsFormat(graph_input_format);
  graph_info_builder.SetInputsDeviceType(graph_input_type);
  graph_info_builder.SetOutputsFormat(graph_output_format);
  graph_info_builder.SetOutputsDeviceType(graph_output_type);

  auto graph_selected_info = graph_info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(graph_selected_info, node.get());
}

void CallbackImpl::SetBasicNodeKernelInfo(const AnfNodePtr &node, const std::vector<inner::NodeBase> &outputs_info) {
  node->set_kernel_info(std::make_shared<device::KernelInfo>());
  std::vector<std::string> input_formats;
  std::vector<TypeId> input_types;
  auto cnode = node->cast<CNodePtr>();
  if (cnode != nullptr) {
    auto &inputs = cnode->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      CollectInputTypesAndFormats(inputs[i], &input_types, &input_formats, true);
    }
  }

  std::vector<std::string> output_formats;
  std::vector<TypeId> output_types;
  AbstractBasePtrList abs_list;
  bool has_fake_abstract = false;
  for (size_t i = 0; i < outputs_info.size(); ++i) {
    output_formats.push_back(outputs_info[i].format);
    output_types.push_back(outputs_info[i].type);
    ShapeVector abs_shape;
    if (outputs_info[i].format != kOpFormat_DEFAULT) {
      abs_shape = GetFakeAbstractShape(outputs_info[i].shape, outputs_info[i].format);
      has_fake_abstract = true;
    } else {
      abs_shape = outputs_info[i].shape;
    }
    auto abs_tensor = std::make_shared<abstract::AbstractTensor>(TypeIdToType(outputs_info[i].type), abs_shape);
    abs_list.push_back(abs_tensor);
  }
  if (has_fake_abstract) {
    if (abs_list.size() == 1) {
      node->set_abstract(abs_list[0]);
    } else {
      node->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
    }
  }

  kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
  info_builder.SetInputsFormat(input_formats);
  info_builder.SetInputsDeviceType(input_types);
  info_builder.SetOutputsFormat(output_formats);
  info_builder.SetOutputsDeviceType(output_types);
  info_builder.SetProcessor(kernel::GetProcessorFromContext());
  info_builder.SetKernelType(KernelType::AKG_KERNEL);
  info_builder.SetFusionType(kernel::kPatternOpaque);
  auto selected_info = info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(selected_info, node.get());
}

void CallbackImpl::SetEmptyKernelInfo(const AnfNodePtr &node) {
  node->set_kernel_info(std::make_shared<device::KernelInfo>());
}

void CallbackImpl::ResetKernelInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (GetTargetFromContext() == kAscendDevice) {
    auto kernel_info = cnode->kernel_info_ptr();
    if (kernel_info == nullptr) {
      cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
    }
    auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kAscendDevice);
    MS_EXCEPTION_IF_NULL(kernel_info_setter);
    kernel_info_setter->SetKernelInfo(cnode, KernelType::UNKNOWN_KERNEL_TYPE);
  } else if (GetTargetFromContext() == kGPUDevice) {
    cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
    auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kGPUDevice);
    MS_EXCEPTION_IF_NULL(kernel_info_setter);
    kernel_info_setter->SetKernelInfo(cnode, KernelType::UNKNOWN_KERNEL_TYPE);
  } else {
    cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
    auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kCPUDevice);
    if (kernel_info_setter != nullptr) {
      kernel_info_setter->SetKernelInfo(cnode, KernelType::UNKNOWN_KERNEL_TYPE);
    }
  }
}

ShapeVector CallbackImplWithInferShape::GetInputShape(const AnfNodePtr &node, size_t i) {
  return CallbackImpl::GetInputInferShape(node, i);
}

ShapeVector CallbackImplWithInferShape::GetOutputShape(const AnfNodePtr &node, size_t i) {
  return common::AnfAlgo::GetOutputInferShape(node, i);
}

TypeId CallbackImplWithInferShape::GetInputType(const AnfNodePtr &node, size_t i) {
  return CallbackImpl::GetInputInferType(node, i);
}

TypeId CallbackImplWithInferShape::GetOutputType(const AnfNodePtr &node, size_t i) {
  return CallbackImpl::GetOutputInferType(node, i);
}

std::string CallbackImplWithInferShape::GetInputFormat(const AnfNodePtr &, size_t) { return kOpFormat_DEFAULT; }

std::string CallbackImplWithInferShape::GetOutputFormat(const AnfNodePtr &, size_t) { return kOpFormat_DEFAULT; }

void CallbackImplWithInferShape::SetBasicNodeKernelInfo(const AnfNodePtr &node,
                                                        const std::vector<inner::NodeBase> &outputs_info) {
  node->set_kernel_info(std::make_shared<device::KernelInfo>());
  if (node->cast<CNodePtr>() != nullptr) {
    return;
  }
  bool has_fake_abstract = false;
  std::vector<TypeId> output_types;
  std::vector<std::string> output_formats;
  AbstractBasePtrList abs_list;
  for (size_t i = 0; i < outputs_info.size(); ++i) {
    output_types.push_back(outputs_info[i].type);
    output_formats.push_back(outputs_info[i].format);
    ShapeVector abs_shape;
    if (outputs_info[i].format != kOpFormat_DEFAULT) {
      abs_shape = GetFakeAbstractShape(outputs_info[i].shape, outputs_info[i].format);
      has_fake_abstract = true;
    } else {
      abs_shape = outputs_info[i].shape;
    }
    abs_list.push_back(std::make_shared<abstract::AbstractTensor>(TypeIdToType(outputs_info[i].type), abs_shape));
  }
  if (has_fake_abstract) {
    if (abs_list.size() == 1) {
      node->set_abstract(abs_list[0]);
    } else {
      node->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
    }
  }

  kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
  info_builder.SetOutputsFormat(output_formats);
  info_builder.SetOutputsDeviceType(output_types);
  AnfAlgo::SetSelectKernelBuildInfo(info_builder.Build(), node.get());
}

std::string CallbackImplWithInferShape::GetProcessor(const AnfNodePtr &) {
  return kernel::GetStrProcessorFromContext();
}
}  // namespace mindspore::graphkernel
