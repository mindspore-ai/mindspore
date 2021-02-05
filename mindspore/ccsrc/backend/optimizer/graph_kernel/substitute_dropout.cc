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
#include "backend/optimizer/graph_kernel/substitute_dropout.h"

#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "base/core_ops.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/tensor.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "runtime/device/kernel_info.h"

namespace mindspore {
namespace opt {
unsigned int SubstituteDropout::seed_ = time(NULL);

const BaseRef SubstituteDropout::DefinePattern() const {
  VarPtr Xs = std::make_shared<Var>();
  return VectorRef({prim::kPrimDropout, Xs});
}

void SetNewKernelInfo(const CNodePtr &kernel_node) {
  std::vector<std::string> inputs_format;
  std::vector<TypeId> inputs_type;
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    inputs_format.emplace_back(AnfAlgo::GetPrevNodeOutputFormat(kernel_node, input_index));
    inputs_type.push_back(AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_index));
  }
  std::vector<std::string> outputs_format;
  std::vector<TypeId> outputs_type;
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    outputs_format.emplace_back(AnfAlgo::GetPrevNodeOutputFormat(kernel_node, output_index));
    outputs_type.push_back(AnfAlgo::GetOutputInferDataType(kernel_node, output_index));
  }
  std::string origin_data_format = kOpFormat_DEFAULT;
  auto cnode_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  cnode_info_builder->SetOriginDataFormat(origin_data_format);
  cnode_info_builder->SetInputsFormat(inputs_format);
  cnode_info_builder->SetInputsDeviceType(inputs_type);
  cnode_info_builder->SetOutputsFormat(outputs_format);
  cnode_info_builder->SetOutputsDeviceType(outputs_type);
  cnode_info_builder->SetKernelType(KernelType::UNKNOWN_KERNEL_TYPE);
  cnode_info_builder->SetProcessor(kernel::Processor::CUDA);
  auto cnode_selected_info = cnode_info_builder->Build();
  AnfAlgo::SetSelectKernelBuildInfo(cnode_selected_info, kernel_node.get());
}

const AnfNodePtr SubstituteDropout::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  CheckCNodeInputSize(cnode, kDropoutInputTensorNum);
  AbstractBasePtr old_abstract = cnode->abstract()->Clone();
  auto shape = AnfAlgo::GetInputDeviceShape(cnode, 0);
  ShapeVector shape_i64;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_i64), [](size_t x) { return SizeToLong(x); });

  // The primitive should use a clone, otherwise the attr seed will be overridden.
  AnfNodePtrList uniform_input = {NewValueNode(prim::kPrimCudnnUniformReal->Clone())};
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, ShapeVector(1, SizeToLong(shape.size())),
                                                 static_cast<void *>(&shape[0]), kNumberTypeInt64);
  uniform_input.push_back(NewValueNode(tensor));
  uniform_input[1]->set_abstract(tensor->ToAbstract());
  uniform_input[1]->set_kernel_info(std::make_shared<device::KernelInfo>());
  std::string origin_data_format = kOpFormat_DEFAULT;
  std::vector<std::string> outputs_format = {origin_data_format};
  std::vector<TypeId> outputs_type = {kNumberTypeInt32};
  auto tensor_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  tensor_info_builder->SetOriginDataFormat(origin_data_format);
  tensor_info_builder->SetOutputsFormat(outputs_format);
  tensor_info_builder->SetOutputsDeviceType(outputs_type);
  tensor_info_builder->SetKernelType(KernelType::UNKNOWN_KERNEL_TYPE);
  tensor_info_builder->SetProcessor(kernel::Processor::CUDA);
  auto tensor_selected_info = tensor_info_builder->Build();
  AnfAlgo::SetSelectKernelBuildInfo(tensor_selected_info, uniform_input[1].get());

  // create new uniform_real_node
  auto uniform_real_node = func_graph->NewCNode(uniform_input);
  SetNodeAttrSafely("seed", MakeValue(SizeToLong(seed_++)), uniform_real_node);
  SetNodeAttrSafely("seed2", MakeValue(SizeToLong(seed_++)), uniform_real_node);
  auto uniform_abstract = std::make_shared<abstract::AbstractTensor>(std::make_shared<Float>(32), shape_i64);
  uniform_real_node->set_abstract(uniform_abstract);
  uniform_real_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  SetNewKernelInfo(uniform_real_node);

  // create new_node, has two input, first is cnode->input[1], second is unifom_real_node
  AnfNodePtrList new_node_inputs = {NewValueNode(prim::kPrimGkDropout)};
  new_node_inputs.push_back(cnode->input(1));
  new_node_inputs.push_back(uniform_real_node);
  auto new_node = func_graph->NewCNode(new_node_inputs);
  SetNodeAttrSafely("keep_prob", MakeValue(AnfAlgo::GetNodeAttr<float>(cnode, "keep_prob")), new_node);
  new_node->set_abstract(old_abstract);
  new_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  SetNewKernelInfo(new_node);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
