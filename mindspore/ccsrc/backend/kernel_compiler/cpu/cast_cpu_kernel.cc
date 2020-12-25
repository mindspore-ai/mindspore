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
  if (thread_num < 1) {
    MS_LOG(ERROR) << "Invalid value: thread_num " << thread_num;
    return;
  }
  threads.reserve(thread_num);
  size_t start = 0;
  size_t once_compute_size = (lens + thread_num - 1) / thread_num;
  if (once_compute_size < 1) {
    MS_LOG(ERROR) << "Invalid value: once_compute_size " << once_compute_size;
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
  mode_map[kNumberTypeBool][kNumberTypeFloat16] = LaunchCast<bool, float16>;
  mode_map[kNumberTypeBool][kNumberTypeFloat32] = LaunchCast<bool, float>;
  mode_map[kNumberTypeBool][kNumberTypeFloat64] = LaunchCast<bool, double>;
  mode_map[kNumberTypeBool][kNumberTypeInt8] = LaunchCast<bool, int8_t>;
  mode_map[kNumberTypeBool][kNumberTypeInt16] = LaunchCast<bool, int16_t>;
  mode_map[kNumberTypeBool][kNumberTypeInt32] = LaunchCast<bool, int32_t>;
  mode_map[kNumberTypeBool][kNumberTypeInt64] = LaunchCast<bool, int64_t>;
  mode_map[kNumberTypeBool][kNumberTypeUInt8] = LaunchCast<bool, uint8_t>;
  mode_map[kNumberTypeBool][kNumberTypeUInt16] = LaunchCast<bool, uint16_t>;
  mode_map[kNumberTypeBool][kNumberTypeUInt32] = LaunchCast<bool, uint32_t>;
  mode_map[kNumberTypeBool][kNumberTypeUInt64] = LaunchCast<bool, uint64_t>;
  mode_map[kNumberTypeBool][kNumberTypeBool] = LaunchCast<bool, bool>;

  mode_map[kNumberTypeFloat16][kNumberTypeFloat16] = LaunchCast<float16, float16>;
  mode_map[kNumberTypeFloat16][kNumberTypeFloat32] = LaunchCast<float16, float>;
  mode_map[kNumberTypeFloat16][kNumberTypeFloat64] = LaunchCast<float16, double>;
  mode_map[kNumberTypeFloat16][kNumberTypeInt8] = LaunchCast<float16, int8_t>;
  mode_map[kNumberTypeFloat16][kNumberTypeInt16] = LaunchCast<float16, int16_t>;
  mode_map[kNumberTypeFloat16][kNumberTypeInt32] = LaunchCast<float16, int32_t>;
  mode_map[kNumberTypeFloat16][kNumberTypeInt64] = LaunchCast<float16, int64_t>;
  mode_map[kNumberTypeFloat16][kNumberTypeUInt8] = LaunchCast<float16, uint8_t>;
  mode_map[kNumberTypeFloat16][kNumberTypeUInt16] = LaunchCast<float16, uint16_t>;
  mode_map[kNumberTypeFloat16][kNumberTypeUInt32] = LaunchCast<float16, uint32_t>;
  mode_map[kNumberTypeFloat16][kNumberTypeUInt64] = LaunchCast<float16, uint64_t>;
  mode_map[kNumberTypeFloat16][kNumberTypeBool] = LaunchCast<float16, bool>;

  mode_map[kNumberTypeFloat32][kNumberTypeFloat16] = LaunchCast<float, float16>;
  mode_map[kNumberTypeFloat32][kNumberTypeFloat32] = LaunchCast<float, float>;
  mode_map[kNumberTypeFloat32][kNumberTypeFloat64] = LaunchCast<float, double>;
  mode_map[kNumberTypeFloat32][kNumberTypeInt8] = LaunchCast<float, int8_t>;
  mode_map[kNumberTypeFloat32][kNumberTypeInt16] = LaunchCast<float, int16_t>;
  mode_map[kNumberTypeFloat32][kNumberTypeInt32] = LaunchCast<float, int32_t>;
  mode_map[kNumberTypeFloat32][kNumberTypeInt64] = LaunchCast<float, int64_t>;
  mode_map[kNumberTypeFloat32][kNumberTypeUInt8] = LaunchCast<float, uint8_t>;
  mode_map[kNumberTypeFloat32][kNumberTypeUInt16] = LaunchCast<float, uint16_t>;
  mode_map[kNumberTypeFloat32][kNumberTypeUInt32] = LaunchCast<float, uint32_t>;
  mode_map[kNumberTypeFloat32][kNumberTypeUInt64] = LaunchCast<float, uint64_t>;
  mode_map[kNumberTypeFloat32][kNumberTypeBool] = LaunchCast<float, bool>;

  mode_map[kNumberTypeFloat64][kNumberTypeFloat16] = LaunchCast<double, float16>;
  mode_map[kNumberTypeFloat64][kNumberTypeFloat32] = LaunchCast<double, float>;
  mode_map[kNumberTypeFloat64][kNumberTypeFloat64] = LaunchCast<double, double>;
  mode_map[kNumberTypeFloat64][kNumberTypeInt8] = LaunchCast<double, int8_t>;
  mode_map[kNumberTypeFloat64][kNumberTypeInt16] = LaunchCast<double, int16_t>;
  mode_map[kNumberTypeFloat64][kNumberTypeInt32] = LaunchCast<double, int32_t>;
  mode_map[kNumberTypeFloat64][kNumberTypeInt64] = LaunchCast<double, int64_t>;
  mode_map[kNumberTypeFloat64][kNumberTypeUInt8] = LaunchCast<double, uint8_t>;
  mode_map[kNumberTypeFloat64][kNumberTypeUInt16] = LaunchCast<double, uint16_t>;
  mode_map[kNumberTypeFloat64][kNumberTypeUInt32] = LaunchCast<double, uint32_t>;
  mode_map[kNumberTypeFloat64][kNumberTypeUInt64] = LaunchCast<double, uint64_t>;
  mode_map[kNumberTypeFloat64][kNumberTypeBool] = LaunchCast<double, bool>;

  mode_map[kNumberTypeInt8][kNumberTypeFloat16] = LaunchCast<int8_t, float16>;
  mode_map[kNumberTypeInt8][kNumberTypeFloat32] = LaunchCast<int8_t, float>;
  mode_map[kNumberTypeInt8][kNumberTypeFloat64] = LaunchCast<int8_t, double>;
  mode_map[kNumberTypeInt8][kNumberTypeInt8] = LaunchCast<int8_t, int8_t>;
  mode_map[kNumberTypeInt8][kNumberTypeInt16] = LaunchCast<int8_t, int16_t>;
  mode_map[kNumberTypeInt8][kNumberTypeInt32] = LaunchCast<int8_t, int32_t>;
  mode_map[kNumberTypeInt8][kNumberTypeInt64] = LaunchCast<int8_t, int64_t>;
  mode_map[kNumberTypeInt8][kNumberTypeUInt8] = LaunchCast<int8_t, uint8_t>;
  mode_map[kNumberTypeInt8][kNumberTypeUInt16] = LaunchCast<int8_t, uint16_t>;
  mode_map[kNumberTypeInt8][kNumberTypeUInt32] = LaunchCast<int8_t, uint32_t>;
  mode_map[kNumberTypeInt8][kNumberTypeUInt64] = LaunchCast<int8_t, uint64_t>;
  mode_map[kNumberTypeInt8][kNumberTypeBool] = LaunchCast<int8_t, bool>;

  mode_map[kNumberTypeInt16][kNumberTypeFloat16] = LaunchCast<int16_t, float16>;
  mode_map[kNumberTypeInt16][kNumberTypeFloat32] = LaunchCast<int16_t, float>;
  mode_map[kNumberTypeInt16][kNumberTypeFloat64] = LaunchCast<int16_t, double>;
  mode_map[kNumberTypeInt16][kNumberTypeInt8] = LaunchCast<int16_t, int8_t>;
  mode_map[kNumberTypeInt16][kNumberTypeInt16] = LaunchCast<int16_t, int16_t>;
  mode_map[kNumberTypeInt16][kNumberTypeInt32] = LaunchCast<int16_t, int32_t>;
  mode_map[kNumberTypeInt16][kNumberTypeInt64] = LaunchCast<int16_t, int64_t>;
  mode_map[kNumberTypeInt16][kNumberTypeUInt8] = LaunchCast<int16_t, uint8_t>;
  mode_map[kNumberTypeInt16][kNumberTypeUInt16] = LaunchCast<int16_t, uint16_t>;
  mode_map[kNumberTypeInt16][kNumberTypeUInt32] = LaunchCast<int16_t, uint32_t>;
  mode_map[kNumberTypeInt16][kNumberTypeUInt64] = LaunchCast<int16_t, uint64_t>;
  mode_map[kNumberTypeInt16][kNumberTypeBool] = LaunchCast<int16_t, bool>;

  mode_map[kNumberTypeInt32][kNumberTypeFloat16] = LaunchCast<int32_t, float16>;
  mode_map[kNumberTypeInt32][kNumberTypeFloat32] = LaunchCast<int32_t, float>;
  mode_map[kNumberTypeInt32][kNumberTypeFloat64] = LaunchCast<int32_t, double>;
  mode_map[kNumberTypeInt32][kNumberTypeInt8] = LaunchCast<int32_t, int8_t>;
  mode_map[kNumberTypeInt32][kNumberTypeInt16] = LaunchCast<int32_t, int16_t>;
  mode_map[kNumberTypeInt32][kNumberTypeInt32] = LaunchCast<int32_t, int32_t>;
  mode_map[kNumberTypeInt32][kNumberTypeInt64] = LaunchCast<int32_t, int64_t>;
  mode_map[kNumberTypeInt32][kNumberTypeUInt8] = LaunchCast<int32_t, uint8_t>;
  mode_map[kNumberTypeInt32][kNumberTypeUInt16] = LaunchCast<int32_t, uint16_t>;
  mode_map[kNumberTypeInt32][kNumberTypeUInt32] = LaunchCast<int32_t, uint32_t>;
  mode_map[kNumberTypeInt32][kNumberTypeUInt64] = LaunchCast<int32_t, uint64_t>;
  mode_map[kNumberTypeInt32][kNumberTypeBool] = LaunchCast<int32_t, bool>;

  mode_map[kNumberTypeInt64][kNumberTypeFloat16] = LaunchCast<int64_t, float16>;
  mode_map[kNumberTypeInt64][kNumberTypeFloat32] = LaunchCast<int64_t, float>;
  mode_map[kNumberTypeInt64][kNumberTypeFloat64] = LaunchCast<int64_t, double>;
  mode_map[kNumberTypeInt64][kNumberTypeInt8] = LaunchCast<int64_t, int8_t>;
  mode_map[kNumberTypeInt64][kNumberTypeInt16] = LaunchCast<int64_t, int16_t>;
  mode_map[kNumberTypeInt64][kNumberTypeInt32] = LaunchCast<int64_t, int32_t>;
  mode_map[kNumberTypeInt64][kNumberTypeInt64] = LaunchCast<int64_t, int64_t>;
  mode_map[kNumberTypeInt64][kNumberTypeUInt8] = LaunchCast<int64_t, uint8_t>;
  mode_map[kNumberTypeInt64][kNumberTypeUInt16] = LaunchCast<int64_t, uint16_t>;
  mode_map[kNumberTypeInt64][kNumberTypeUInt32] = LaunchCast<int64_t, uint32_t>;
  mode_map[kNumberTypeInt64][kNumberTypeUInt64] = LaunchCast<int64_t, uint64_t>;
  mode_map[kNumberTypeInt64][kNumberTypeBool] = LaunchCast<int64_t, bool>;

  mode_map[kNumberTypeUInt8][kNumberTypeFloat16] = LaunchCast<uint8_t, float16>;
  mode_map[kNumberTypeUInt8][kNumberTypeFloat32] = LaunchCast<uint8_t, float>;
  mode_map[kNumberTypeUInt8][kNumberTypeFloat64] = LaunchCast<uint8_t, double>;
  mode_map[kNumberTypeUInt8][kNumberTypeInt8] = LaunchCast<uint8_t, int8_t>;
  mode_map[kNumberTypeUInt8][kNumberTypeInt16] = LaunchCast<uint8_t, int16_t>;
  mode_map[kNumberTypeUInt8][kNumberTypeInt32] = LaunchCast<uint8_t, int32_t>;
  mode_map[kNumberTypeUInt8][kNumberTypeInt64] = LaunchCast<uint8_t, int64_t>;
  mode_map[kNumberTypeUInt8][kNumberTypeUInt8] = LaunchCast<uint8_t, uint8_t>;
  mode_map[kNumberTypeUInt8][kNumberTypeUInt16] = LaunchCast<uint8_t, uint16_t>;
  mode_map[kNumberTypeUInt8][kNumberTypeUInt32] = LaunchCast<uint8_t, uint32_t>;
  mode_map[kNumberTypeUInt8][kNumberTypeUInt64] = LaunchCast<uint8_t, uint64_t>;
  mode_map[kNumberTypeUInt8][kNumberTypeBool] = LaunchCast<uint8_t, bool>;

  mode_map[kNumberTypeUInt16][kNumberTypeFloat16] = LaunchCast<uint16_t, float16>;
  mode_map[kNumberTypeUInt16][kNumberTypeFloat32] = LaunchCast<uint16_t, float>;
  mode_map[kNumberTypeUInt16][kNumberTypeFloat64] = LaunchCast<uint16_t, double>;
  mode_map[kNumberTypeUInt16][kNumberTypeInt8] = LaunchCast<uint16_t, int8_t>;
  mode_map[kNumberTypeUInt16][kNumberTypeInt16] = LaunchCast<uint16_t, int16_t>;
  mode_map[kNumberTypeUInt16][kNumberTypeInt32] = LaunchCast<uint16_t, int32_t>;
  mode_map[kNumberTypeUInt16][kNumberTypeInt64] = LaunchCast<uint16_t, int64_t>;
  mode_map[kNumberTypeUInt16][kNumberTypeUInt8] = LaunchCast<uint16_t, uint8_t>;
  mode_map[kNumberTypeUInt16][kNumberTypeUInt16] = LaunchCast<uint16_t, uint16_t>;
  mode_map[kNumberTypeUInt16][kNumberTypeUInt32] = LaunchCast<uint16_t, uint32_t>;
  mode_map[kNumberTypeUInt16][kNumberTypeUInt64] = LaunchCast<uint16_t, uint64_t>;
  mode_map[kNumberTypeUInt16][kNumberTypeBool] = LaunchCast<uint16_t, bool>;

  mode_map[kNumberTypeUInt32][kNumberTypeFloat16] = LaunchCast<uint32_t, float16>;
  mode_map[kNumberTypeUInt32][kNumberTypeFloat32] = LaunchCast<uint32_t, float>;
  mode_map[kNumberTypeUInt32][kNumberTypeFloat64] = LaunchCast<uint32_t, double>;
  mode_map[kNumberTypeUInt32][kNumberTypeInt8] = LaunchCast<uint32_t, int8_t>;
  mode_map[kNumberTypeUInt32][kNumberTypeInt16] = LaunchCast<uint32_t, int16_t>;
  mode_map[kNumberTypeUInt32][kNumberTypeInt32] = LaunchCast<uint32_t, int32_t>;
  mode_map[kNumberTypeUInt32][kNumberTypeInt64] = LaunchCast<uint32_t, int64_t>;
  mode_map[kNumberTypeUInt32][kNumberTypeUInt8] = LaunchCast<uint32_t, uint8_t>;
  mode_map[kNumberTypeUInt32][kNumberTypeUInt16] = LaunchCast<uint32_t, uint16_t>;
  mode_map[kNumberTypeUInt32][kNumberTypeUInt32] = LaunchCast<uint32_t, uint32_t>;
  mode_map[kNumberTypeUInt32][kNumberTypeUInt64] = LaunchCast<uint32_t, uint64_t>;
  mode_map[kNumberTypeUInt32][kNumberTypeBool] = LaunchCast<uint32_t, bool>;

  mode_map[kNumberTypeUInt64][kNumberTypeFloat16] = LaunchCast<uint64_t, float16>;
  mode_map[kNumberTypeUInt64][kNumberTypeFloat32] = LaunchCast<uint64_t, float>;
  mode_map[kNumberTypeUInt64][kNumberTypeFloat64] = LaunchCast<uint64_t, double>;
  mode_map[kNumberTypeUInt64][kNumberTypeInt8] = LaunchCast<uint64_t, int8_t>;
  mode_map[kNumberTypeUInt64][kNumberTypeInt16] = LaunchCast<uint64_t, int16_t>;
  mode_map[kNumberTypeUInt64][kNumberTypeInt32] = LaunchCast<uint64_t, int32_t>;
  mode_map[kNumberTypeUInt64][kNumberTypeInt64] = LaunchCast<uint64_t, int64_t>;
  mode_map[kNumberTypeUInt64][kNumberTypeUInt8] = LaunchCast<uint64_t, uint8_t>;
  mode_map[kNumberTypeUInt64][kNumberTypeUInt16] = LaunchCast<uint64_t, uint16_t>;
  mode_map[kNumberTypeUInt64][kNumberTypeUInt32] = LaunchCast<uint64_t, uint32_t>;
  mode_map[kNumberTypeUInt64][kNumberTypeUInt64] = LaunchCast<uint64_t, uint64_t>;
  mode_map[kNumberTypeUInt64][kNumberTypeBool] = LaunchCast<uint64_t, bool>;

  mode_map[source_dtype][target_dtype](inputs, outputs);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
