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
template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::map<std::pair<TypeId, TypeId>, ContiguousGpuKernel::ContiguousFunc> ContiguousGpuKernel::func_list_ = {
  {std::make_pair(kNumberTypeFloat16, kNumberTypeFloat16), &ContiguousGpuKernel::LaunchContiguousImpl<half>},
  {std::make_pair(kNumberTypeFloat32, kNumberTypeFloat32), &ContiguousGpuKernel::LaunchContiguousImpl<float>},
  {std::make_pair(kNumberTypeFloat64, kNumberTypeFloat64), &ContiguousGpuKernel::LaunchContiguousImpl<double>},
  {std::make_pair(kNumberTypeInt8, kNumberTypeInt8), &ContiguousGpuKernel::LaunchContiguousImpl<int8_t>},
  {std::make_pair(kNumberTypeInt16, kNumberTypeInt16), &ContiguousGpuKernel::LaunchContiguousImpl<int16_t>},
  {std::make_pair(kNumberTypeInt32, kNumberTypeInt32), &ContiguousGpuKernel::LaunchContiguousImpl<int32_t>},
  {std::make_pair(kNumberTypeInt64, kNumberTypeInt64), &ContiguousGpuKernel::LaunchContiguousImpl<int64_t>},
  {std::make_pair(kNumberTypeBool, kNumberTypeBool), &ContiguousGpuKernel::LaunchContiguousImpl<bool>},
  {std::make_pair(kNumberTypeComplex64, kNumberTypeFloat32), &ContiguousGpuKernel::LaunchContiguousImpl<float>},
  {std::make_pair(kNumberTypeComplex128, kNumberTypeFloat64), &ContiguousGpuKernel::LaunchContiguousImpl<double>},
  {std::make_pair(kNumberTypeComplex64, kNumberTypeComplex64),
   &ContiguousGpuKernel::LaunchContiguousImpl<Complex<float>>},
  {std::make_pair(kNumberTypeComplex128, kNumberTypeComplex128),
   &ContiguousGpuKernel::LaunchContiguousImpl<Complex<double>>},
  {std::make_pair(kNumberTypeUInt8, kNumberTypeUInt8), &ContiguousGpuKernel::LaunchContiguousImpl<uint8_t>},
  {std::make_pair(kNumberTypeUInt16, kNumberTypeUInt16), &ContiguousGpuKernel::LaunchContiguousImpl<uint16_t>},
  {std::make_pair(kNumberTypeUInt32, kNumberTypeUInt32), &ContiguousGpuKernel::LaunchContiguousImpl<uint32_t>},
  {std::make_pair(kNumberTypeUInt64, kNumberTypeUInt64), &ContiguousGpuKernel::LaunchContiguousImpl<uint64_t>}};

bool ContiguousGpuKernel::LaunchContiguous(TypeId input_type_id, const kernel::AddressPtr &input,
                                           const TensorStorageInfoPtr &input_storage_info, TypeId output_type_id,
                                           const kernel::AddressPtr &output, void *stream_ptr) {
  const auto &iter = func_list_.find(std::make_pair(input_type_id, output_type_id));
  if (iter == func_list_.end()) {
    MS_LOG(EXCEPTION) << "type_id:" << input_type_id << " is invalid";
  }
  int64_t type_size = GetDataTypeSize(input_type_id) / GetDataTypeSize(output_type_id);

  return iter->second(this, input, input_storage_info, output, type_size, stream_ptr);
}

template <typename T>
bool ContiguousGpuKernel::LaunchContiguousImpl(const kernel::AddressPtr &input,
                                               const TensorStorageInfoPtr &input_storage_info,
                                               const kernel::AddressPtr &output, const int64_t &type_size,
                                               void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(input_storage_info);
  T *input_addr = GetDeviceAddress<T>({input}, 0);
  T *output_addr = GetDeviceAddress<T>({output}, 0);
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
    auto status = CalAsStrided(output_size, input_addr, output_addr, input_storage_info,
                               reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, "Contiguous");
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
