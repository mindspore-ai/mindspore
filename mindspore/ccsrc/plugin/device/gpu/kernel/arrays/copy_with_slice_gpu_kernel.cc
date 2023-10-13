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

#include "plugin/device/gpu/kernel/arrays/copy_with_slice_gpu_kernel.h"
#include <functional>
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "utils/log_adapter.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/copy_with_slice_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::unordered_map<TypeId, CopyWithSliceGpuKernel::CopyWithSliceFunc> CopyWithSliceGpuKernel::func_list_ = {
  {kNumberTypeFloat16, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<half>},
  {kNumberTypeFloat32, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<float>},
  {kNumberTypeFloat64, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<double>},
  {kNumberTypeInt8, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<int8_t>},
  {kNumberTypeInt16, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<int16_t>},
  {kNumberTypeInt32, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<int32_t>},
  {kNumberTypeInt64, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<int64_t>},
  {kNumberTypeBool, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<bool>},
  {kNumberTypeComplex64, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<Complex<float>>},
  {kNumberTypeComplex128, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<Complex<double>>},
  {kNumberTypeUInt8, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<uint8_t>},
  {kNumberTypeUInt16, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<uint16_t>},
  {kNumberTypeUInt32, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<uint32_t>},
  {kNumberTypeUInt64, &CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl<uint64_t>}};

bool CopyWithSliceGpuKernel::LaunchCopyWithSlice(TypeId type_id, const TensorStorageInfoPtr &src_storage_info,
                                                 const kernel::AddressPtr &src_addr,
                                                 const TensorStorageInfoPtr &dst_storage_info,
                                                 const kernel::AddressPtr &dst_addr, void *stream_ptr) {
  const auto &iter = func_list_.find(type_id);
  if (iter == func_list_.end()) {
    MS_LOG(EXCEPTION) << "type_id:" << type_id << " is invalid";
  }

  return iter->second(this, src_storage_info, src_addr, dst_storage_info, dst_addr, stream_ptr);
}

template <typename T>
bool CopyWithSliceGpuKernel::LaunchCopyWithSliceImpl(const TensorStorageInfoPtr &src_storage_info,
                                                     const kernel::AddressPtr &src_addr,
                                                     const TensorStorageInfoPtr &dst_storage_info,
                                                     const kernel::AddressPtr &dst_addr, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(dst_storage_info);
  T *copy_src_addr = GetDeviceAddress<T>({src_addr}, 0);
  T *self_addr = GetDeviceAddress<T>({dst_addr}, 0);
  const auto &output_shape = dst_storage_info->shape;
  auto output_size =
    LongToSize(std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>()));
  auto src_is_contiguous = src_storage_info == nullptr || src_storage_info->is_contiguous;

  if (dst_storage_info->is_contiguous && src_is_contiguous) {
    auto &dst_storage_offset = dst_storage_info->storage_offset;
    if ((dst_storage_offset + output_size) * sizeof(T) > dst_addr->size) {
      MS_LOG(EXCEPTION) << "Offset is out of bounds, offset:" << (dst_storage_offset * sizeof(T))
                        << " output_size:" << (output_size * sizeof(T)) << " dst_addr->size:" << dst_addr->size;
    }
    size_t src_storage_offset{0};
    if (src_storage_info != nullptr) {
      src_storage_offset = src_storage_info->storage_offset;
    }

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(self_addr + dst_storage_offset, copy_src_addr + src_storage_offset, output_size * sizeof(T),
                      cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpy output failed");
  } else {
    auto status = CalCopyWithSlice(output_size, copy_src_addr, src_storage_info, self_addr, dst_storage_info,
                                   reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, "CopyWithSlice");
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
