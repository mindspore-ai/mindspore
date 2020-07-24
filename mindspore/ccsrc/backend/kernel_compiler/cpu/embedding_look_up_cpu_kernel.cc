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
#include <thread>
#include <string>
#include "backend/kernel_compiler/cpu/embedding_look_up_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
namespace {
void LookUpTableTask(const float *input_addr, const int *indices_addr, float *output_addr, size_t indices_lens,
                     size_t outer_dim_size, int offset, size_t first_dim_size) {
  size_t lens = outer_dim_size * sizeof(float);
  for (size_t i = 0; i < indices_lens; ++i) {
    int index = indices_addr[i] - offset;
    if (index >= 0 && index < SizeToInt(first_dim_size)) {
      size_t pos = index * outer_dim_size;
      auto ret = memcpy_s(output_addr, lens, input_addr + pos, lens);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "LookUpTable task memcpy failed.";
      }
    } else {
      auto ret = memset_s(output_addr, lens, 0, lens);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "LookUpTable task memset failed.";
      }
    }
    output_addr += outer_dim_size;
  }
}
}  // namespace

void EmbeddingLookUpCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  std::vector<size_t> input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.empty()) {
    MS_LOG(EXCEPTION) << "param must be at least 1D";
  }
  first_dim_size_ = input_shape[0];
  for (size_t i = 1; i < input_shape.size(); ++i) {
    outer_dim_size_ *= input_shape[i];
  }
  std::vector<size_t> indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  for (const auto &shape : indices_shape) {
    indices_lens_ *= shape;
  }
  if (AnfAlgo::HasNodeAttr(kAttrOffset, kernel_node)) {
    offset_ = AnfAlgo::GetNodeAttr<int>(kernel_node, kAttrOffset);
  }
}

bool EmbeddingLookUpCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> & /*workspace*/,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto indices_addr = reinterpret_cast<int *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  const size_t thread_num = 8;
  std::thread threads[8];
  size_t task_proc_lens = (indices_lens_ + thread_num - 1) / thread_num;
  size_t i;
  size_t task_offset = 0;
  MS_LOG(DEBUG) << "indices_lens_: " << indices_lens_ << " one task proc lens:" << task_proc_lens;
  for (i = 0; i < thread_num; i++) {
    if (task_offset >= indices_lens_) {
      break;
    }
    MS_LOG(DEBUG) << "task_offset: " << task_offset << " task_proc_lenss:" << task_proc_lens;
    threads[i] =
      std::thread(LookUpTableTask, input_addr, indices_addr + task_offset, output_addr + task_offset * outer_dim_size_,
                  task_proc_lens, outer_dim_size_, offset_, first_dim_size_);
    task_offset += task_proc_lens;
    if (task_offset + task_proc_lens > indices_lens_) {
      task_proc_lens = indices_lens_ - task_offset;
    }
  }
  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  return true;
}

void EmbeddingLookUpCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > 4) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size()
                      << ", but EmbeddingLookUpCPUKernel only support 4d or lower.";
  }

  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but EmbeddingLookUpCPUKernel needs 2.";
  }
}
}  // namespace kernel
}  // namespace mindspore
