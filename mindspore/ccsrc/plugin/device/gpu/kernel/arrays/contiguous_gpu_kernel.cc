/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/contiguous_gpu_kernel.h"
#include <functional>
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "utils/log_adapter.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/as_strided_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"

namespace mindspore {
namespace kernel {
std::unordered_map<TypeId, ContiguousGpuKernel::ContiguousFunc> ContiguousGpuKernel::func_list_ = {
  {kNumberTypeFloat16, &ContiguousGpuKernel::LaunchContiguousImpl<half>},
  {kNumberTypeFloat32, &ContiguousGpuKernel::LaunchContiguousImpl<float>},
  {kNumberTypeFloat64, &ContiguousGpuKernel::LaunchContiguousImpl<double>},
  {kNumberTypeInt8, &ContiguousGpuKernel::LaunchContiguousImpl<int8_t>},
  {kNumberTypeInt16, &ContiguousGpuKernel::LaunchContiguousImpl<int16_t>},
  {kNumberTypeInt32, &ContiguousGpuKernel::LaunchContiguousImpl<int32_t>},
  {kNumberTypeInt64, &ContiguousGpuKernel::LaunchContiguousImpl<int64_t>},
  {kNumberTypeBool, &ContiguousGpuKernel::LaunchContiguousImpl<bool>},
  {kNumberTypeComplex64, &ContiguousGpuKernel::LaunchContiguousImpl<float>},
  {kNumberTypeComplex128, &ContiguousGpuKernel::LaunchContiguousImpl<double>},
  {kNumberTypeUInt8, &ContiguousGpuKernel::LaunchContiguousImpl<uint8_t>},
  {kNumberTypeUInt16, &ContiguousGpuKernel::LaunchContiguousImpl<uint16_t>},
  {kNumberTypeUInt32, &ContiguousGpuKernel::LaunchContiguousImpl<uint32_t>},
  {kNumberTypeUInt64, &ContiguousGpuKernel::LaunchContiguousImpl<uint64_t>}};

bool ContiguousGpuKernel::LaunchContiguous(TypeId type_id, const kernel::AddressPtr &input,
                                           const TensorStorageInfoPtr &input_storage_info,
                                           const kernel::AddressPtr &output, void *stream_ptr) {
  const auto &iter = func_list_.find(type_id);
  if (iter == func_list_.end()) {
    MS_LOG(EXCEPTION) << "type_id:" << type_id << " is invalid";
  }
  bool is_complex = (type_id == kNumberTypeComplex64 || type_id == kNumberTypeComplex128);

  return iter->second(this, input, input_storage_info, output, is_complex, stream_ptr);
}

template <typename T>
bool ContiguousGpuKernel::LaunchContiguousImpl(const kernel::AddressPtr &input,
                                               const TensorStorageInfoPtr &input_storage_info,
                                               const kernel::AddressPtr &output, bool is_complex, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(input_storage_info);
  T *input_addr = GetDeviceAddress<T>({input}, 0);
  T *output_addr = GetDeviceAddress<T>({output}, 0);
  int64_t type_size = is_complex ? 2 : 1;
  const auto &output_shape = input_storage_info->shape;
  auto output_size =
    LongToSize(std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>()));
  output_size *= type_size;
  if (input_storage_info->is_contiguous) {
    auto &offset = input_storage_info->storage_offset;
    auto input_size = input->size * type_size;
    if ((offset + output_size) * sizeof(T) > input_size) {
      MS_LOG(EXCEPTION) << "Offset is out of bounds, offset:" << (offset * sizeof(T))
                        << " output_size:" << (output_size * sizeof(T)) << " input->size:" << input_size;
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_addr, input_addr + offset, output_size * sizeof(T), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpy output failed");
  } else {
    CalAsStrided(output_size, input_addr, output_addr, input_storage_info, reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
