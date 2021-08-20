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
#include "backend/kernel_compiler/cpu/gather_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "nnacl/gather_parameter.h"
#include "nnacl/base/gather_base.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
template <typename T>
void GatherV2CPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  indices_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (!is_dynamic_shape_) {
    axis_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  }
}

template <typename T>
bool GatherV2CPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  auto input_tensor = reinterpret_cast<int8_t *>(inputs[0]->addr);
  indices_data_ = reinterpret_cast<int32_t *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<int8_t *>(outputs[0]->addr);
  if (is_dynamic_shape_) {
    axis_ = reinterpret_cast<int64_t *>(inputs[2]->addr)[0];
  }

  int dims = SizeToInt(input_shape_.size());
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(ERROR) << "axis must be in the range [-rank, rank)";
    return false;
  } else if (axis_ < 0) {
    axis_ = axis_ + dims;
  }

  int max_thread_num = static_cast<int>(common::ThreadPool::GetInstance().GetSyncRunThreadNum());
  ParallelRun(input_tensor, output_addr, max_thread_num);
  return true;
}

template <typename T>
void GatherV2CPUKernel<T>::ParallelRun(int8_t *input_addr, int8_t *output_addr, int thread_num) {
  int outer_size = 1, inner_size = 1;
  for (int64_t i = 0; i < axis_; ++i) {
    outer_size *= input_shape_.at(i);
  }
  for (size_t i = axis_ + 1; i < input_shape_.size(); ++i) {
    inner_size *= input_shape_.at(i);
  }
  int indices_element_size = 1;
  for (size_t i = 0; i < indices_shape_.size(); i++) {
    indices_element_size *= indices_shape_.at(i);
  }
  const int limit = input_shape_.at(axis_);
  int stride = UP_DIV(outer_size, thread_num);
  std::vector<common::Task> tasks;
  int thread_index = 0;
  while (thread_index < thread_num) {
    int count = MSMIN(stride, outer_size - stride * thread_index);
    if (count <= 0) break;
    auto thread_stride = stride * thread_index;
    int8_t *in = input_addr + thread_stride * limit * inner_size * sizeof(T);
    int8_t *out = output_addr + thread_stride * indices_element_size * inner_size * sizeof(T);
    auto block = [&, in, count, out]() {
      int ret = Gather(in, count, inner_size, limit, indices_data_, indices_element_size, out, sizeof(T));
      if (ret != 0) {
        MS_LOG(ERROR) << "GatherRun error task_id[" << thread_index << "] error_code[" << ret << "]";
        return common::FAIL;
      }
      return common::SUCCESS;
    };
    tasks.emplace_back(block);
    thread_index++;
  }
  if (!common::ThreadPool::GetInstance().SyncRun(tasks)) {
    MS_LOG(EXCEPTION) << "SyncRun error!";
  }
}

template <typename T>
void GatherV2CPUKernel<T>::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num == 3) {
    is_dynamic_shape_ = true;
    MS_LOG(DEBUG) << " GatherV2CPUKernel running in Dynamic Mode.";
  } else if (input_num == 2) {
    MS_LOG(DEBUG) << " GatherV2CPUKernel running in Normal Mode.";
  } else {
    MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but GatherV2CPUKernel needs 2.";
  }
}
}  // namespace kernel
}  // namespace mindspore
