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
namespace prim {
inline const PrimitivePtr kPrimGkDropout = std::make_shared<Primitive>("GkDropout");
}  // namespace prim
namespace opt {
int64_t DropoutExpander::seed_ = time(nullptr);

AnfNodePtr DropoutExpander::PreProcess(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  CheckCNodeInputSize(cnode, kDropoutInputTensorNum);
  auto shape = AnfAlgo::GetInputDeviceShape(cnode, 0);
  ShapeVector shape_i64;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_i64), [](size_t x) { return SizeToLong(x); });

  // Create a uniform_real kernel to generate random value.
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, ShapeVector(1, SizeToLong(shape.size())),
                                                 static_cast<void *>(&shape[0]), kNumberTypeInt64);
  AnfNodePtrList uniform_real_input = {NewValueNode(prim::kPrimCudnnUniformReal), NewValueNode(tensor)};
  uniform_real_input[1]->set_abstract(tensor->ToAbstract());
  uniform_real_input[1]->set_kernel_info(std::make_shared<device::KernelInfo>());
  auto uniform_real_node = func_graph->NewCNode(uniform_real_input);
  SetNodeAttrSafely("seed", MakeValue(seed_++), uniform_real_node);
  AnfAlgo::SetNodeAttr("seed2", MakeValue(seed_++), uniform_real_node);
  uniform_real_node->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, shape_i64));
  // Set kernel_info for uniform_real node
  auto uniform_real_kernel_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  uniform_real_kernel_info_builder->SetInputsFormat({kOpFormat_DEFAULT});
  uniform_real_kernel_info_builder->SetInputsDeviceType({kNumberTypeInt32});
  uniform_real_kernel_info_builder->SetOutputsFormat({kOpFormat_DEFAULT});
  uniform_real_kernel_info_builder->SetOutputsDeviceType({kNumberTypeFloat32});
  uniform_real_kernel_info_builder->SetKernelType(KernelType::UNKNOWN_KERNEL_TYPE);
  uniform_real_kernel_info_builder->SetProcessor(kernel::Processor::CUDA);
  AnfAlgo::SetSelectKernelBuildInfo(uniform_real_kernel_info_builder->Build(), uniform_real_node.get());

  // Create a GKDropout node with uniform_real as its second input.
  AnfNodePtrList gkdropout_inputs = {NewValueNode(prim::kPrimGkDropout), cnode->input(1), uniform_real_node};
  auto new_dropout_node = func_graph->NewCNode(gkdropout_inputs);
  SetNodeAttrSafely("keep_prob", MakeValue(AnfAlgo::GetNodeAttr<float>(cnode, "keep_prob")), new_dropout_node);
  // the output info is unchanged.
  new_dropout_node->set_abstract(node->abstract());
  auto old_kernel_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  auto dropout_kernel_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(old_kernel_info);
  dropout_kernel_info_builder->SetInputsFormat({old_kernel_info->GetInputFormat(0), kOpFormat_DEFAULT});
  dropout_kernel_info_builder->SetInputsDeviceType({old_kernel_info->GetInputDeviceType(0), kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(dropout_kernel_info_builder->Build(), new_dropout_node.get());
  return new_dropout_node;
}

AnfNodePtr DropoutExpander::Run(const AnfNodePtr &node) {
  auto gkdropout_node = PreProcess(node->func_graph(), node);
  return DefaultExpander::Run(gkdropout_node);
}
}  // namespace opt
}  // namespace mindspore
