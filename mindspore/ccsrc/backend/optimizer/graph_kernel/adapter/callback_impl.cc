/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/graph_kernel/adapter/callback_impl.h"

#include <algorithm>
#include <vector>
#include <utility>
#include <memory>
#include "utils/ms_context.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/adapter/fake_abstract_shape.h"

namespace mindspore::graphkernel {
// register the callback object
GRAPH_KERNEL_CALLBACK_REGISTER(CallbackImpl);

ShapeVector CallbackImpl::GetInputShape(const AnfNodePtr &node, size_t i) {
  auto vec = AnfAlgo::GetInputDeviceShape(node, i);
  ShapeVector ret;
  std::transform(vec.begin(), vec.end(), std::back_inserter(ret), SizeToLong);
  return ret;
}

ShapeVector CallbackImpl::GetOutputShape(const AnfNodePtr &node, size_t i) {
  auto vec = AnfAlgo::GetOutputDeviceShape(node, i);
  ShapeVector ret;
  std::transform(vec.begin(), vec.end(), std::back_inserter(ret), SizeToLong);
  return ret;
}

ShapeVector CallbackImpl::GetInputInferShape(const AnfNodePtr &node, size_t i) {
  auto vec = AnfAlgo::GetPrevNodeOutputInferShape(node, i);
  ShapeVector ret;
  std::transform(vec.begin(), vec.end(), std::back_inserter(ret), SizeToLong);
  return ret;
}

ShapeVector CallbackImpl::GetOutputInferShape(const AnfNodePtr &node, size_t i) {
  auto vec = AnfAlgo::GetOutputInferShape(node, i);
  ShapeVector ret;
  std::transform(vec.begin(), vec.end(), std::back_inserter(ret), SizeToLong);
  return ret;
}

TypeId CallbackImpl::GetInputType(const AnfNodePtr &node, size_t i) { return AnfAlgo::GetInputDeviceDataType(node, i); }

TypeId CallbackImpl::GetOutputType(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetOutputDeviceDataType(node, i);
}

TypeId CallbackImpl::GetInputInferType(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetPrevNodeOutputInferDataType(node, i);
}

TypeId CallbackImpl::GetOutputInferType(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetOutputInferDataType(node, i);
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
    auto kernel_with_index = AnfUtils::VisitKernel(inputs[i], 0);
    if (kernel_with_index.first->isa<ValueNode>()) {
      auto tensor = GetValueNode<tensor::TensorPtr>(kernel_with_index.first);
      MS_EXCEPTION_IF_NULL(tensor);
      (void)graph_input_format.emplace_back(kOpFormat_DEFAULT);
      (void)graph_input_type.emplace_back(tensor->data_type());
    } else {
      auto input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
      (void)graph_input_format.emplace_back(std::move(input_format));
      auto input_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
      (void)graph_input_type.emplace_back(input_type);
    }
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
    auto kernel_with_index = AnfAlgo::VisitKernel(outputs[i], 0);
    auto output_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
    graph_output_format.push_back(output_format);
    graph_output_type.push_back(output_type);
  }
  kernel::KernelBuildInfo::KernelBuildInfoBuilder graph_info_builder;
  graph_info_builder.SetInputsFormat(graph_input_format);
  graph_info_builder.SetInputsDeviceType(graph_input_type);
  graph_info_builder.SetOutputsFormat(graph_output_format);
  graph_info_builder.SetOutputsDeviceType(graph_output_type);
  graph_info_builder.SetProcessor(kernel::GetProcessorFromContext());
  graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  graph_info_builder.SetFusionType(kernel::FusionType::OPAQUE);
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
      auto kernel_with_index = AnfAlgo::VisitKernel(inputs[i], 0);
      auto input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
      input_formats.push_back(input_format);
      auto input_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
      input_types.push_back(input_type);
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
  info_builder.SetFusionType(kernel::FusionType::OPAQUE);
  auto selected_info = info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(selected_info, node.get());
}

void CallbackImpl::SetEmptyKernelInfo(const AnfNodePtr &node) {
  node->set_kernel_info(std::make_shared<device::KernelInfo>());
}
}  // namespace mindspore::graphkernel
