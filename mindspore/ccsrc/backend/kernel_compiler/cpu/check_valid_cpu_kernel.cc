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

#include <functional>

#include "backend/kernel_compiler/cpu/check_valid_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputSize = 2;
constexpr size_t kOutputSize = 1;
}  // namespace

template <typename T>
void CheckValidCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  anchor_box_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  img_metas_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
}

template <typename T>
bool CheckValidCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CheckParams(inputs, outputs);
  auto anchor_box = reinterpret_cast<T *>(inputs[0]->addr);
  auto img_metas = reinterpret_cast<T *>(inputs[1]->addr);
  auto output = reinterpret_cast<bool *>(outputs[0]->addr);
  const size_t elem_num = inputs[0]->size / sizeof(T) / COORDINATE;

  auto task = [this, &anchor_box, &img_metas, &output](size_t start, size_t end) {
    const T ZERO = static_cast<T>(0);
    const T ONE = static_cast<T>(1);
    constexpr size_t OFFSET_ZERO = 0;
    constexpr size_t OFFSET_ONE = 1;
    constexpr size_t OFFSET_TWO = 2;
    for (size_t i = start; i < end; i++) {
      const size_t left_x = i * 4;
      const size_t left_y = i * 4 + 1;
      const size_t right_x = i * 4 + 2;
      const size_t right_y = i * 4 + 3;

      bool valid_flag = false;
      valid_flag = valid_flag || std::less<T>()(anchor_box[left_x], ZERO);
      valid_flag = valid_flag || std::less<T>()(anchor_box[left_y], ZERO);
      valid_flag =
        valid_flag || std::less<T>()(img_metas[OFFSET_ONE] * img_metas[OFFSET_TWO] - ONE, anchor_box[right_x]);
      valid_flag =
        valid_flag || std::less<T>()(img_metas[OFFSET_ZERO] * img_metas[OFFSET_TWO] - ONE, anchor_box[right_y]);

      output[i] = !valid_flag;
    }
  };
  CPUKernelUtils::ParallelFor(task, elem_num);

  return true;
}

template <typename T>
void CheckValidCPUKernel<T>::CheckParams(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  //  inputs: anchor_box, img_metas
  if (inputs.size() != kInputSize) {
    MS_LOG(EXCEPTION) << "Input number is: " << inputs.size() << ", but CheckValid needs " << kInputSize << " inputs.";
  }

  //  outputs: valid
  if (outputs.size() != kOutputSize) {
    MS_LOG(EXCEPTION) << "Output number is: " << outputs.size() << ", but CheckValid needs " << kOutputSize
                      << "outputs.";
  }
  if (outputs[0]->size / sizeof(bool) != inputs[0]->size / sizeof(T) / COORDINATE) {
    MS_LOG(EXCEPTION) << "The output dimensions must match the dimensions of img_metas.";
  }
}
}  // namespace kernel
}  // namespace mindspore
