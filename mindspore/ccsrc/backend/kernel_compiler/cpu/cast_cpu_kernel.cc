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
#include <cmath>
#include <map>
#include <string>
#include "backend/kernel_compiler/cpu/cast_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename S, typename T>
void Cast(const S *in, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(in[i]);
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename S, typename T>
void CastCPUKernel<S, T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  source_dtype = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, 0);
  target_dtype = AnfAlgo::GetOutputInferDataType(kernel_node, 0);
}

template <typename S, typename T>
bool CastCPUKernel<S, T>::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  S *input = reinterpret_cast<S *>(inputs[0]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  MS_LOG(DEBUG) << "Type source: " << typeid(S).name() << "; target: " << typeid(T).name();

  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  Cast<S, T>(input, output, lens);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
