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

#include "backend/kernel_compiler/cpu/cummax_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void CummaxCPUKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output1_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  output2_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 1);
  dim_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "dim");
}

template <typename T>
bool CummaxCPUKernelMod<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  auto input_data_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output1_data_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto output2_data_addr = reinterpret_cast<int64_t *>(outputs[1]->addr);

  const size_t dims = input_shape_.size();
  if (dims == 0) {
    MS_LOG(EXCEPTION) << "The value of `dims` can not be 0";
  }
  dim_ = (dim_ + dims) % dims;
  std::vector<size_t> p{1};

  for (int64_t i = (int64_t)input_shape_.size() - 1; i >= 0; i--)
    p.push_back(p[(int64_t)input_shape_.size() - 1 - i] * input_shape_[i]);
  reverse(p.begin(), p.end());

  size_t input_stride = p[dim_ + 1];
  size_t output1_stride = p[dim_ + 1];
  size_t output2_stride = p[dim_ + 1];
  size_t input_dim_size = input_shape_[dim_];

  int exit_ok = 0;
  std::vector<size_t> counter(dims, 0);

  while (!exit_ok) {
    T out = input_data_addr[0];
    int idx = 0;
    for (size_t i = 0; i < input_dim_size; i++) {
      T cur = input_data_addr[i * input_stride];
      if (cur >= out) {
        out = cur;
        idx = i;
      }
      output1_data_addr[i * output1_stride] = out;
      output2_data_addr[i * output2_stride] = idx;
    }

    if (dims == 1) break;
    for (size_t dim_i = 0; dim_i < dims; dim_i++) {
      if (dim_i == dim_) {
        if (dim_i == dims - 1) {
          exit_ok = 1;
          break;
        }
        continue;
      }
      counter[dim_i]++;
      input_data_addr += p[dim_i + 1];
      output1_data_addr += p[dim_i + 1];
      output2_data_addr += p[dim_i + 1];

      if (counter[dim_i] == input_shape_[dim_i]) {
        if (dim_i == dims - 1) {
          exit_ok = 1;
          break;
        } else {
          input_data_addr -= counter[dim_i] * p[dim_i + 1];
          output1_data_addr -= counter[dim_i] * p[dim_i + 1];
          output2_data_addr -= counter[dim_i] * p[dim_i + 1];
          counter[dim_i] = 0;
        }
      } else {
        break;
      }
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
