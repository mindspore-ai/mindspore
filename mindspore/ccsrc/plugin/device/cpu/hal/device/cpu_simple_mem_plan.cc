/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/hal/device/cpu_simple_mem_plan.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace device {
namespace cpu {
size_t CPUSimpleMemPlan::MemPlan(const session::KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  size_t total_mem_size = 32;
  auto kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
    for (size_t i = 0; i < input_num; ++i) {
      auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel, i);
      MS_EXCEPTION_IF_NULL(kernel_with_index.first);
      if (kernel_with_index.first->isa<Parameter>()) {
        continue;
      }
      auto address = AnfAlgo::GetOutputAddr(kernel_with_index.first, kernel_with_index.second, true);
      MS_EXCEPTION_IF_NULL(address);
      if (address->ptr_ == nullptr) {
        total_mem_size += address->size_;
      }
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel);
    for (size_t i = 0; i < output_num; ++i) {
      auto address = AnfAlgo::GetOutputAddr(kernel, i);
      MS_EXCEPTION_IF_NULL(address);
      if (address->ptr_ == nullptr) {
        total_mem_size += address->size_;
      }
    }

    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
      auto address = AnfAlgo::GetWorkspaceAddr(kernel, i);
      MS_EXCEPTION_IF_NULL(address);
      if (address->ptr_ == nullptr) {
        total_mem_size += address->size_;
      }
    }
  }

  return total_mem_size;
}

void CPUSimpleMemPlan::MemAssign(const session::KernelGraph *graph, uint8_t *base_ptr) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(base_ptr);
  uint8_t *mem_ptr = base_ptr;
  auto kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
    for (size_t i = 0; i < input_num; ++i) {
      auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel, i);
      MS_EXCEPTION_IF_NULL(kernel_with_index.first);
      if (kernel_with_index.first->isa<Parameter>()) {
        continue;
      }
      auto address = AnfAlgo::GetMutableOutputAddr(kernel_with_index.first, kernel_with_index.second, true);
      MS_EXCEPTION_IF_NULL(address);
      if (address->ptr_ == nullptr) {
        address->ptr_ = mem_ptr;
        mem_ptr = mem_ptr + address->size_;
      }
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel);
    for (size_t i = 0; i < output_num; ++i) {
      auto address = AnfAlgo::GetMutableOutputAddr(kernel, i);
      MS_EXCEPTION_IF_NULL(address);
      if (address->ptr_ == nullptr) {
        address->ptr_ = mem_ptr;
        mem_ptr = mem_ptr + address->size_;
      }
    }

    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
      auto address = AnfAlgo::GetWorkspaceAddr(kernel, i);
      MS_EXCEPTION_IF_NULL(address);
      if (address->ptr_ == nullptr) {
        address->ptr_ = mem_ptr;
        mem_ptr = mem_ptr + address->size_;
      }
    }
  }
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
