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

#include "runtime/device/launch_mul.h"
#include "abstract/utils.h"
#include "backend/common/session/single_kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore::device {
std::shared_ptr<session::KernelGraph> LaunchMul::ObtainMulKernelGraph() const {
  std::vector<TypeId> input_dtypes = {dtype_, dtype_};
  std::vector<TypeId> output_dtypes = {dtype_};
  // obtain input & output shapes
  size_t dtype_size = abstract::TypeIdSize(dtype_);
  if (dtype_size == 0) {
    MS_LOG(EXCEPTION) << "Divide by zero.";
  }
  int64_t shape = SizeToLong(total_size_ / dtype_size);
  std::vector<std::vector<int64_t>> input_shapes = {{shape}, {1}};
  std::vector<ShapeVector> output_shapes = {{shape}};
  auto mul_graph = session::SingleKernelGraph::ConstructKernelGraphBasedOnSingleOp(
    kMulOpName, input_dtypes, input_shapes, output_dtypes, output_shapes);
  MS_EXCEPTION_IF_NULL(mul_graph);
  return mul_graph;
}

kernel::KernelMod *LaunchMul::ObtainLaunchMulKernelMod() {
  if (mul_graph_ == nullptr) {
    // construct mul kernel graph
    mul_graph_ = ObtainMulKernelGraph();
    MS_EXCEPTION_IF_NULL(mul_graph_);
    // kernel select
    KernelSelect(mul_graph_);
    // kernel build
    KernelBuild(mul_graph_);
  }
  // obtain kernel_mod
  if (mul_graph_->execution_order().size() != 1) {
    MS_LOG(ERROR) << "the execution order of the mul graph should have only one node, however, it has "
                  << mul_graph_->execution_order().size() << " nodes.";
  }
  return AnfAlgo::GetKernelMod(mul_graph_->execution_order()[0]);
}

void LaunchMul::ObtainMulInputsAddr() {
  inputs_addr_.push_back(input1_addr_);

  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto device_num = parallel_context->device_num();
  if (device_num == 0) {
    MS_LOG(ERROR) << "device num can't be zero";
  }
  input2_value_ = 1.0f / device_num;
  auto size = abstract::TypeIdSize(dtype_);
  auto input_size = AlignSizeForLaunchKernel(size * 1);
  // alloc memory
  input2_addr_ = AllocDeviceMem(input_size);
  CopyHostMemToDevice(size, input_size);
  inputs_addr_.push_back(input2_addr_);
}

void LaunchMul::FreeInputDeviceMemory() {
  input1_addr_ = nullptr;
  if (input2_addr_ != nullptr) {
    FreeDeviceMem(input2_addr_);
    input2_addr_ = nullptr;
  }
  inputs_addr_.clear();
}
}  // namespace mindspore::device
