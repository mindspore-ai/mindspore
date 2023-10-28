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

#include <complex>
#include "plugin/device/cpu/kernel/copy_with_slice_cpu_kernel.h"

#include "utils/log_adapter.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
std::unordered_map<TypeId, CopyWithSliceCpuKernel::CopyWithSliceFunc> CopyWithSliceCpuKernel::func_list_ = {
  {kNumberTypeFloat16, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<float16>},
  {kNumberTypeFloat32, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<float>},
  {kNumberTypeFloat64, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<double>},
  {kNumberTypeInt8, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<int8_t>},
  {kNumberTypeInt16, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<int16_t>},
  {kNumberTypeInt32, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<int32_t>},
  {kNumberTypeInt64, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<int64_t>},
  {kNumberTypeBool, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<bool>},
  {kNumberTypeComplex64, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<complex64>},
  {kNumberTypeComplex128, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<complex128>},
  {kNumberTypeUInt8, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<uint8_t>},
  {kNumberTypeUInt16, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<uint16_t>},
  {kNumberTypeUInt32, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<uint32_t>},
  {kNumberTypeUInt64, &CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl<uint64_t>}};

bool CopyWithSliceCpuKernel::LaunchCopyWithSlice(TypeId type_id, const TensorStorageInfoPtr &src_storage_info,
                                                 const kernel::AddressPtr &src_addr,
                                                 const TensorStorageInfoPtr &dst_storage_info,
                                                 const kernel::AddressPtr &dst_addr) {
  const auto &iter = func_list_.find(type_id);
  if (iter == func_list_.end()) {
    MS_LOG(EXCEPTION) << "type_id:" << type_id << " is invalid";
  }

  return iter->second(this, src_storage_info, src_addr, dst_storage_info, dst_addr);
}

int64_t OffsetCalc(size_t ndim, const ShapeVector &shape, int64_t tmp_index, const std::vector<int64_t> &strides) {
  int64_t offset = 0;
  for (size_t dim = 0; dim < ndim; dim++) {
    auto index = ndim - dim - 1;
    int64_t mod = tmp_index % shape[index];
    tmp_index = tmp_index / shape[index];
    offset += mod * strides[index];
  }
  return offset;
}

template <typename T>
bool CopyWithSliceCpuKernel::LaunchCopyWithSliceImpl(const TensorStorageInfoPtr &src_storage_info,
                                                     const kernel::AddressPtr &src_addr,
                                                     const TensorStorageInfoPtr &dst_storage_info,
                                                     const kernel::AddressPtr &dst_addr) {
  MS_EXCEPTION_IF_NULL(dst_storage_info);
  T *copy_src_addr = GetDeviceAddress<T>({src_addr}, 0);
  T *self_addr = GetDeviceAddress<T>({dst_addr}, 0);
  MS_EXCEPTION_IF_NULL(copy_src_addr);
  MS_EXCEPTION_IF_NULL(self_addr);
  const auto &output_shape = dst_storage_info->shape;
  auto output_size =
    LongToSize(std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>()));
  auto &dst_storage_offset = dst_storage_info->storage_offset;
  size_t src_storage_offset{0};
  if (src_storage_info != nullptr) {
    src_storage_offset = src_storage_info->storage_offset;
  }
  auto src_is_contiguous = src_storage_info == nullptr || src_storage_info->is_contiguous;

  if (dst_storage_info->is_contiguous && src_is_contiguous) {
    if ((dst_storage_offset + output_size) * sizeof(T) > dst_addr->size) {
      MS_LOG(EXCEPTION) << "Offset is out of bounds, offset:" << (dst_storage_offset * sizeof(T))
                        << " output_size:" << (output_size * sizeof(T)) << " dst_addr->size:" << dst_addr->size;
    }

    int ret = memcpy_s(self_addr + dst_storage_offset, output_size * sizeof(T), copy_src_addr + src_storage_offset,
                       output_size * sizeof(T));
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
    }
  } else if (!dst_storage_info->is_contiguous && src_is_contiguous) {
    size_t dst_ndim = dst_storage_info->shape.size();
    auto dst_strides = dst_storage_info->strides;

    auto task = [&](size_t start, size_t end) {
      for (size_t pos = start; pos < end; ++pos) {
        size_t dst_offset = LongToSize(OffsetCalc(dst_ndim, output_shape, pos, dst_strides));
        self_addr[dst_offset + dst_storage_offset] = copy_src_addr[pos + src_storage_offset];
      }
    };
    ParallelLaunch(task, output_size, 0, this, pool_);
  } else if (dst_storage_info->is_contiguous && !src_is_contiguous) {
    const auto &input_shape = src_storage_info->shape;
    size_t src_ndim = src_storage_info->shape.size();
    auto src_strides = src_storage_info->strides;

    auto task = [&](size_t start, size_t end) {
      for (size_t pos = start; pos < end; ++pos) {
        size_t src_offset = LongToSize(OffsetCalc(src_ndim, input_shape, pos, src_strides));
        self_addr[pos + dst_storage_offset] = copy_src_addr[src_offset + src_storage_offset];
      }
    };
    ParallelLaunch(task, output_size, 0, this, pool_);
  } else {
    size_t dst_ndim = dst_storage_info->shape.size();
    auto dst_strides = dst_storage_info->strides;
    const auto &input_shape = src_storage_info->shape;
    size_t src_ndim = src_storage_info->shape.size();
    auto src_strides = src_storage_info->strides;

    auto task = [&](size_t start, size_t end) {
      for (size_t pos = start; pos < end; ++pos) {
        size_t dst_offset = LongToSize(OffsetCalc(dst_ndim, output_shape, pos, dst_strides));
        size_t src_offset = LongToSize(OffsetCalc(src_ndim, input_shape, pos, src_strides));
        self_addr[dst_offset + dst_storage_offset] = copy_src_addr[src_offset + src_storage_offset];
      }
    };
    ParallelLaunch(task, output_size, 0, this, pool_);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
