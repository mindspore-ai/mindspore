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

#include "plugin/device/ascend/hal/device/ascend_launch_atomic_clean.h"
#include "abstract/utils.h"
#include "backend/common/session/single_kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore::device::ascend {
void AscendLaunchAtomicClean::FreeDeviceMem(void *addr) { AscendLaunchKernel::FreeDeviceMem(addr); }

size_t AscendLaunchAtomicClean::AlignSizeForLaunchKernel(size_t size) {
  return AscendLaunchKernel::AlignSizeForLaunchKernel(size);
}

uint8_t *AscendLaunchAtomicClean::AllocDeviceMem(size_t size) { return AscendLaunchKernel::AllocDeviceMem(size); }

void AscendLaunchAtomicClean::KernelSelect(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  AscendLaunchKernel::KernelSelect(kernel_graph);
}

void AscendLaunchAtomicClean::KernelBuild(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  AscendLaunchKernel::KernelBuild(kernel_graph);
}

void AscendLaunchAtomicClean::LaunchOpKernel() {
  if (atomic_clean_graph_ == nullptr) {
    // construct atomic clean kernel graph and set attr
    ConstructKernelGraphAndSetAttr();
    // kernel build
    KernelBuild(atomic_clean_graph_);
  }
  // obtain kernel_mod
  if (atomic_clean_graph_->execution_order().size() != 1) {
    MS_LOG(ERROR) << "The execution order of the atomic clean graph should have only one node";
  }
  kernel_mod_ = AnfAlgo::GetKernelMod(atomic_clean_graph_->execution_order()[0]);
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  // obtain kernel inputs
  std::vector<kernel::AddressPtr> kernel_inputs;
  auto input = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(input);
  input->addr = input_addr_;
  MS_EXCEPTION_IF_NULL(input->addr);
  input->size = IntToSize(total_size_);
  kernel_inputs.push_back(input);
  // obtain kernel outputs
  auto kernel_outputs = ObtainKernelOutputs(kernel_mod_->GetOutputSizeList());
  // obtain kernel workspace
  auto kernel_workspaces = ObtainKernelWorkspaces(kernel_mod_->GetWorkspaceSizeList());
  // launch
  auto ret_status = kernel_mod_->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, stream_);
  if (!ret_status) {
    MS_LOG(ERROR) << "Launch single kernel failed.";
  }
}

void AscendLaunchAtomicClean::FreeLaunchDeviceMem() {
  input_addr_ = nullptr;
  FreeOutputAndWorkspaceDeviceMem();
}

std::shared_ptr<session::KernelGraph> AscendLaunchAtomicClean::ObtainAtomicCleanKernelGraph() {
  std::vector<TypeId> input_dtypes = {dtype_};
  std::vector<TypeId> output_dtypes = {};
  // obtain input & output shapes
  size_t dtype_size = abstract::TypeIdSize(dtype_);
  if (dtype_size == 0) {
    MS_LOG(EXCEPTION) << "Divide by zero.";
  }
  auto shape = total_size_ / dtype_size;
  std::vector<std::vector<int64_t>> input_shapes = {{static_cast<int64_t>(shape)}};
  std::vector<ShapeVector> output_shapes = {};
  auto atomic_clean_graph = session::SingleKernelGraph::ConstructKernelGraphBasedOnSingleOp(
    kAtomicAddrCleanOpName, input_dtypes, input_shapes, output_dtypes, output_shapes);
  MS_EXCEPTION_IF_NULL(atomic_clean_graph);
  return atomic_clean_graph;
}

void AscendLaunchAtomicClean::ConstructKernelGraphAndSetAttr() {
  // construct atomic clean kernel graph
  atomic_clean_graph_ = ObtainAtomicCleanKernelGraph();
  MS_EXCEPTION_IF_NULL(atomic_clean_graph_);
  // set atomic clean attr
  if (!atomic_clean_graph_->execution_order().empty()) {
    auto clean_node = atomic_clean_graph_->execution_order()[0];
    // set abstract
    AbstractBasePtr abstract = std::make_shared<abstract::AbstractNone>();
    MS_EXCEPTION_IF_NULL(clean_node);
    clean_node->set_abstract(abstract);
    // set build info
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(builder);
    builder->SetKernelType(KernelType::TBE_KERNEL);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), clean_node.get());
    // set attr
    std::vector<int32_t> clean_size = {total_size_};
    common::AnfAlgo::SetNodeAttr(kAttrAtomicAddMemSize, MakeValue(clean_size), clean_node);
  }
}
}  // namespace mindspore::device::ascend
