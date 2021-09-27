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

#include "backend/kernel_compiler/cpu/assign_cpu_kernel.h"

#include <string>
#include <map>

#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAssignInputsNum = 2;
constexpr size_t kAssignOutputsNum = 1;

const std::map<TypeId, size_t> input_x_dtype_size_map = {
  {kNumberTypeBool, sizeof(bool)},       {kNumberTypeInt8, sizeof(int8_t)},     {kNumberTypeInt16, sizeof(int16_t)},
  {kNumberTypeInt32, sizeof(int32_t)},   {kNumberTypeInt64, sizeof(int64_t)},   {kNumberTypeUInt8, sizeof(uint8_t)},
  {kNumberTypeUInt16, sizeof(uint16_t)}, {kNumberTypeUInt32, sizeof(uint32_t)}, {kNumberTypeUInt64, sizeof(uint64_t)},
  {kNumberTypeFloat16, sizeof(float16)}, {kNumberTypeFloat32, sizeof(float)},   {kNumberTypeFloat64, sizeof(double)}};
}  // namespace

void AssignCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  auto input_x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto input_y_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (input_x_shape.size() != input_y_shape.size()) {
    MS_LOG(EXCEPTION) << "X and y must be same shape!";
  }
  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    if (input_x_shape[i] != input_y_shape[i]) {
      MS_LOG(EXCEPTION) << "X and y must be same shape!";
    }
    batch_size_ *= input_x_shape[i];
  }
  input_x_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto type_len = input_x_dtype_size_map.find(input_x_dtype_);
  if (type_len == input_x_dtype_size_map.end()) {
    MS_LOG(EXCEPTION) << "Unsupported input_x dtype!";
  }
  input_x_dtype_size_ = type_len->second;
}

bool AssignCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                             const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAssignInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAssignOutputsNum, kernel_name_);
  auto max_size = inputs[0]->size;
  size_t total_size = input_x_dtype_size_ * batch_size_;
  if (total_size > max_size) {
    MS_LOG(EXCEPTION) << "Memcpy size must <= max_size, but got memcpy size is : " << total_size
                      << ", max size is : " << max_size;
  }
  constexpr size_t kBlockSize = 10000;
  size_t thread_num = (total_size + kBlockSize - 1) / kBlockSize;
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  thread_num = thread_num > max_thread_num ? max_thread_num : thread_num;
  if (thread_num == 0) {
    return true;
  }
  size_t stride = total_size / thread_num;
  std::vector<common::Task> tasks;
  size_t thread_index = 0;
  auto input0_addr = reinterpret_cast<int8_t *>(inputs[0]->addr);
  auto input1_addr = reinterpret_cast<int8_t *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<int8_t *>(outputs[0]->addr);
  size_t length = stride;
  while (thread_index < thread_num) {
    auto thread_stride = stride * thread_index;
    size_t max_length = total_size - thread_stride;
    if (thread_index == thread_num - 1) {
      length = max_length;
    }
    int8_t *input0 = input0_addr + thread_stride;
    int8_t *input1 = input1_addr + thread_stride;
    int8_t *output = output_addr + thread_stride;
    auto block = [input0, input1, output, max_length, length]() {
      int ret = memcpy_s(input0, max_length, input1, length);
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, error no " << ret;
        return common::FAIL;
      }
      ret = memcpy_s(output, max_length, input1, length);
      if (ret != 0) {
        MS_LOG(ERROR) << "memcpy_s error, error no " << ret;
        return common::FAIL;
      }
      return common::SUCCESS;
    };
    (void)tasks.emplace_back(block);
    thread_index++;
  }
  (void)common::ThreadPool::GetInstance().SyncRun(tasks);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
