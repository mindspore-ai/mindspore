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
#include <thread>
#include "backend/kernel_compiler/cpu/cast_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename S, typename T>
void Cast(const S *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = static_cast<T>(in[i]);
  }
}

template <typename S, typename T>
void LaunchCast(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) {
  S *input = reinterpret_cast<S *>(inputs[0]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  MS_LOG(DEBUG) << "Type source: " << typeid(S).name() << "; target: " << typeid(T).name();

  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  auto max_thread_num = std::thread::hardware_concurrency();
  size_t thread_num = lens < 128 * max_thread_num ? std::ceil(lens / 128.0) : max_thread_num;
  MS_LOG(INFO) << "Lens=" << lens << "; use thread_num=" << thread_num << "; max_thread_num: " << max_thread_num;
  std::vector<std::thread> threads;
  threads.reserve(thread_num);
  size_t start = 0;
  size_t once_compute_size = (lens + thread_num - 1) / thread_num;
  if (thread_num < 1 || once_compute_size < 1) {
    MS_LOG(ERROR) << "Invalid value: thread_num " << thread_num << "; once_compute_size " << once_compute_size;
    return;
  }
  while (start < lens) {
    size_t end = (start + once_compute_size) > lens ? lens : (start + once_compute_size);
    threads.emplace_back(std::thread(Cast<S, T>, input, output, start, end));
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

void CastCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  source_dtype = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, 0);
  target_dtype = AnfAlgo::GetOutputInferDataType(kernel_node, 0);
}

bool CastCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> & /*workspace*/,
                           const std::vector<kernel::AddressPtr> &outputs) {
  using TypePair =
    std::function<void(const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  std::map<TypeId, std::map<TypeId, TypePair>> mode_map;
  mode_map[kNumberTypeFloat32][kNumberTypeFloat32] = LaunchCast<float, float>;
  mode_map[kNumberTypeFloat32][kNumberTypeInt32] = LaunchCast<float, int>;
  mode_map[kNumberTypeFloat32][kNumberTypeBool] = LaunchCast<float, bool>;
  mode_map[kNumberTypeInt32][kNumberTypeFloat32] = LaunchCast<int, float>;
  mode_map[kNumberTypeInt32][kNumberTypeInt32] = LaunchCast<int, int>;
  mode_map[kNumberTypeInt32][kNumberTypeBool] = LaunchCast<int, bool>;
  mode_map[kNumberTypeBool][kNumberTypeFloat32] = LaunchCast<bool, float>;
  mode_map[kNumberTypeBool][kNumberTypeBool] = LaunchCast<bool, bool>;
  mode_map[kNumberTypeBool][kNumberTypeInt32] = LaunchCast<bool, int>;
  mode_map[source_dtype][target_dtype](inputs, outputs);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
