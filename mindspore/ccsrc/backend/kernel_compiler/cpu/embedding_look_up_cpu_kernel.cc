/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/embedding_look_up_cpu_kernel.h"
#include <thread>
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"
#include "ir/primitive.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBlockSize = 10000;
constexpr size_t kEmbeddingLookupInputsNum = 2;
constexpr size_t kEmbeddingLookupOutputsNum = 1;
constexpr size_t kEmbeddingLookupInputParamsMaxDim = 2;

template <typename T>
void LookUpTableTask(const float *input_addr, const T *indices_addr, float *output_addr, size_t indices_lens,
                     size_t outer_dim_size, T offset, size_t first_dim_size) {
  auto type_size = sizeof(float);
  size_t lens = outer_dim_size * type_size;
  for (size_t i = 0; i < indices_lens; ++i) {
    T index = indices_addr[i] - offset;
    if (index >= 0 && index < SizeToInt(first_dim_size)) {
      size_t pos = static_cast<size_t>(index) * outer_dim_size;
      auto ret = memcpy_s(output_addr, (indices_lens - i) * lens, input_addr + pos, lens);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "LookUpTable task memcpy failed.";
      }
    } else {
      auto ret = memset_s(output_addr, (indices_lens - i) * lens, 0, lens);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "LookUpTable task memset failed.";
      }
    }
    output_addr += outer_dim_size;
  }
}
}  // namespace

void EmbeddingLookUpCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.empty() || input_shape.size() > kEmbeddingLookupInputParamsMaxDim) {
    MS_LOG(EXCEPTION) << "EmbeddingLookUpCPUKernel support 1-" << kEmbeddingLookupInputParamsMaxDim
                      << "D input tensor, but got " << input_shape.size() << "D.";
  }
  first_dim_size_ = input_shape[0];
  outer_dim_size_ = 1;
  for (size_t i = 1; i < input_shape.size(); ++i) {
    outer_dim_size_ *= input_shape[i];
  }
  indices_lens_ = 1;
  std::vector<size_t> indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  for (const auto &shape : indices_shape) {
    indices_lens_ *= shape;
  }
  indices_data_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  if (AnfAlgo::HasNodeAttr(kAttrOffset, kernel_node)) {
    offset_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrOffset);
  }
}

template <typename T>
void EmbeddingLookUpCPUKernel::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  if (!node_wpt_.expired()) {
    auto node = node_wpt_.lock();
    if (!node) {
      MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
    }
    std::vector<size_t> input_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
    if (input_shape.empty()) {
      MS_LOG(EXCEPTION) << "Param must be at least 1D";
    }
    first_dim_size_ = input_shape[0];
    outer_dim_size_ = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) {
      outer_dim_size_ *= input_shape[i];
    }

    indices_lens_ = 1;
    std::vector<size_t> indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
    for (const auto &shape : indices_shape) {
      indices_lens_ *= shape;
    }
  }
  const auto *input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  const auto *indices_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto *output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  size_t thread_num = indices_lens_ / kBlockSize + 1;
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  thread_num = thread_num > max_thread_num ? max_thread_num : thread_num;
  std::vector<common::Task> tasks;
  size_t task_proc_lens = (indices_lens_ + thread_num - 1) / thread_num;
  size_t i;
  size_t task_offset = 0;
  MS_LOG(DEBUG) << "indices_lens_: " << indices_lens_ << " one task proc lens:" << task_proc_lens;
  for (i = 0; i < thread_num; i++) {
    if (task_offset >= indices_lens_) {
      break;
    }
    MS_LOG(DEBUG) << "task_offset: " << task_offset << " task_proc_lenss:" << task_proc_lens;
    auto task = [input_addr, indices_addr, output_addr, task_offset, task_proc_lens, this]() {
      LookUpTableTask<T>(input_addr, indices_addr + task_offset, output_addr + task_offset * outer_dim_size_,
                         task_proc_lens, outer_dim_size_, static_cast<T>(offset_), first_dim_size_);
      return common::SUCCESS;
    };
    (void)tasks.emplace_back(task);
    task_offset += task_proc_lens;
    if (task_offset + task_proc_lens > indices_lens_) {
      task_proc_lens = indices_lens_ - task_offset;
    }
  }
  (void)common::ThreadPool::GetInstance().SyncRun(tasks);
}

bool EmbeddingLookUpCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kEmbeddingLookupInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kEmbeddingLookupOutputsNum, kernel_name_);
  if (indices_data_type_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else {
    LaunchKernel<int64_t>(inputs, outputs);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
