/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
}  // namespace

template <typename T>
void CheckValidCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  anchor_box_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  img_metas_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
}

template <typename T>
bool CheckValidCpuKernelMod<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CheckParams(inputs, outputs);
  auto anchor_box = reinterpret_cast<T *>(inputs[0]->addr);
  auto img_metas = reinterpret_cast<T *>(inputs[1]->addr);
  auto output = reinterpret_cast<bool *>(outputs[0]->addr);
  const size_t elem_num = inputs[0]->size / sizeof(T) / COORDINATE;

  const double offset = 1.0;
  auto height = static_cast<double>(img_metas[kIndex0]);
  auto width = static_cast<double>(img_metas[kIndex1]);
  auto ratio = static_cast<double>(img_metas[kIndex2]);
  auto img_width_x = width * ratio - offset;
  auto img_height_y = height * ratio - offset;

  auto task = [this, &anchor_box, &img_width_x, &img_height_y, &output](size_t start, size_t end) {
    const T ZERO = static_cast<T>(0);
    for (size_t i = start; i < end; i++) {
      const size_t left_x = i * 4;
      const size_t left_y = i * 4 + 1;
      const size_t right_x = i * 4 + 2;
      const size_t right_y = i * 4 + 3;

      bool valid_flag = false;
      valid_flag |= std::less<T>()(anchor_box[left_x], ZERO);
      valid_flag |= std::less<T>()(anchor_box[left_y], ZERO);
      valid_flag |= std::less<double>()(img_width_x, static_cast<double>(anchor_box[right_x]));
      valid_flag |= std::less<double>()(img_height_y, static_cast<double>(anchor_box[right_y]));

      output[i] = !valid_flag;
    }
  };
  ParallelLaunchAutoSearch(task, elem_num, this, &parallel_search_info_);
  return true;
}

template <typename T>
void CheckValidCpuKernelMod<T>::CheckParams(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &outputs) {
  //  inputs: anchor_box, img_metas
  if (inputs.size() != kInputSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << kInputSize << ", but got "
                      << inputs.size();
  }

  //  outputs: valid
  if (outputs.size() != kOutputSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be " << kOutputSize << ", but got "
                      << outputs.size();
  }
  if (outputs[0]->size / sizeof(bool) != inputs[0]->size / sizeof(T) / COORDINATE) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output should be the same as 'img_metas', but got the shape of output: "
                      << Vector2Str(output_shape_) << ", the shape of 'img_metas': " << Vector2Str(img_metas_shape_);
  }
}
}  // namespace kernel
}  // namespace mindspore
