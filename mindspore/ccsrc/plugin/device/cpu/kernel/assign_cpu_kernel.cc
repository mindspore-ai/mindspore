/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/assign_cpu_kernel.h"

#include <string>
#include <map>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
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

void AssignCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  auto input_x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto input_y_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (input_x_shape.size() != input_y_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'x' and 'y' should have the same dimension, but got the dimension of 'x': "
                      << input_x_shape.size() << " and the dimension of 'y': " << input_y_shape.size();
  }
  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    if (input_x_shape[i] != input_y_shape[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the 'x' and 'y' should have the same shape, but got the shape of 'x': "
                        << Vector2Str(input_x_shape) << " and the shape of 'y': " << Vector2Str(input_y_shape);
    }
    batch_size_ *= input_x_shape[i];
  }
  input_x_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto type_len = input_x_dtype_size_map.find(input_x_dtype_);
  if (type_len == input_x_dtype_size_map.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' should be bool, int, uint, or float, but got " << input_x_dtype_;
  }
  input_x_dtype_size_ = type_len->second;
}

bool AssignCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAssignInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAssignOutputsNum, kernel_name_);
  static std::string kernel_name = kernel_name_;
  auto max_size = inputs[0]->size;
  size_t total_size = input_x_dtype_size_ * batch_size_;
  if (total_size > max_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', memcpy size must be less than or equal to max size, but got memcpy size: " << total_size
                      << ", and max size: " << max_size;
  }

  auto input0_addr = reinterpret_cast<int8_t *>(inputs[0]->addr);
  auto input1_addr = reinterpret_cast<int8_t *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<int8_t *>(outputs[0]->addr);
  auto task = [&](size_t start, size_t end) {
    int8_t *input0 = input0_addr + start;
    int8_t *input1 = input1_addr + start;
    int8_t *output = output_addr + start;
    size_t length = end - start;
    size_t max_length = total_size - start;
    int ret = memcpy_s(input0, max_length, input1, length);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', memcpy_s error. Error no " << ret;
    }
    ret = memcpy_s(output, max_length, input1, length);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', memcpy_s error. Error no " << ret;
    }
  };
  ParallelLaunchAutoSearch(task, total_size, this, &parallel_search_info_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
