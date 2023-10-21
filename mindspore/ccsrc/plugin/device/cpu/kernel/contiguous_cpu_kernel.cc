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

#include "plugin/device/cpu/kernel/contiguous_cpu_kernel.h"
#include <functional>
#include <complex>
#include "utils/log_adapter.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
std::map<std::pair<TypeId, TypeId>, ContiguousCpuKernel::ContiguousFunc> ContiguousCpuKernel::func_list_ = {
  {std::make_pair(kNumberTypeFloat16, kNumberTypeFloat16), &ContiguousCpuKernel::LaunchContiguousImpl<float16>},
  {std::make_pair(kNumberTypeFloat32, kNumberTypeFloat32), &ContiguousCpuKernel::LaunchContiguousImpl<float>},
  {std::make_pair(kNumberTypeFloat64, kNumberTypeFloat64), &ContiguousCpuKernel::LaunchContiguousImpl<double>},
  {std::make_pair(kNumberTypeInt8, kNumberTypeInt8), &ContiguousCpuKernel::LaunchContiguousImpl<int8_t>},
  {std::make_pair(kNumberTypeInt16, kNumberTypeInt16), &ContiguousCpuKernel::LaunchContiguousImpl<int16_t>},
  {std::make_pair(kNumberTypeInt32, kNumberTypeInt32), &ContiguousCpuKernel::LaunchContiguousImpl<int32_t>},
  {std::make_pair(kNumberTypeInt64, kNumberTypeInt64), &ContiguousCpuKernel::LaunchContiguousImpl<int64_t>},
  {std::make_pair(kNumberTypeBool, kNumberTypeBool), &ContiguousCpuKernel::LaunchContiguousImpl<bool>},
  {std::make_pair(kNumberTypeComplex64, kNumberTypeFloat32), &ContiguousCpuKernel::LaunchContiguousImpl<float>},
  {std::make_pair(kNumberTypeComplex128, kNumberTypeFloat64), &ContiguousCpuKernel::LaunchContiguousImpl<double>},
  {std::make_pair(kNumberTypeComplex64, kNumberTypeComplex64), &ContiguousCpuKernel::LaunchContiguousImpl<complex64>},
  {std::make_pair(kNumberTypeComplex128, kNumberTypeComplex128),
   &ContiguousCpuKernel::LaunchContiguousImpl<complex128>},
  {std::make_pair(kNumberTypeUInt8, kNumberTypeUInt8), &ContiguousCpuKernel::LaunchContiguousImpl<uint8_t>},
  {std::make_pair(kNumberTypeUInt16, kNumberTypeUInt16), &ContiguousCpuKernel::LaunchContiguousImpl<uint16_t>},
  {std::make_pair(kNumberTypeUInt32, kNumberTypeUInt32), &ContiguousCpuKernel::LaunchContiguousImpl<uint32_t>},
  {std::make_pair(kNumberTypeUInt64, kNumberTypeUInt64), &ContiguousCpuKernel::LaunchContiguousImpl<uint64_t>}};

bool ContiguousCpuKernel::LaunchContiguous(TypeId input_type_id, const kernel::AddressPtr &input,
                                           const TensorStorageInfoPtr &input_storage_info, TypeId output_type_id,
                                           const kernel::AddressPtr &output) {
  const auto &iter = func_list_.find(std::make_pair(input_type_id, output_type_id));
  if (iter == func_list_.end()) {
    MS_LOG(EXCEPTION) << "type_id:" << input_type_id << " is invalid";
  }
  int64_t type_size = SizeToLong(GetDataTypeSize(input_type_id) / GetDataTypeSize(output_type_id));

  return iter->second(this, input, input_storage_info, output, type_size);
}

template <typename T>
bool ContiguousCpuKernel::LaunchContiguousImpl(const kernel::AddressPtr &input,
                                               const TensorStorageInfoPtr &input_storage_info,
                                               const kernel::AddressPtr &output, const int64_t &type_size) {
  MS_EXCEPTION_IF_NULL(input_storage_info);
  T *input_addr = GetDeviceAddress<T>({input}, 0);
  T *output_addr = GetDeviceAddress<T>({output}, 0);
  MS_EXCEPTION_IF_NULL(input_addr);
  MS_EXCEPTION_IF_NULL(output_addr);
  const auto &output_shape = input_storage_info->shape;
  auto output_size =
    LongToSize(std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>()));
  if (output_size == 0) {
    // CPU unsupported zero copy
    return true;
  }
  output_size *= type_size;
  if (input_storage_info->is_contiguous) {
    auto &offset = input_storage_info->storage_offset;
    auto ret = memcpy_s(output_addr, output_size * sizeof(T), input_addr + offset, output_size * sizeof(T));
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "For 'ConvertToDynamic', memcpy_s error. Error no: " << ret
                        << " ,output_addr:" << output_addr << " size=" << output_size * sizeof(T)
                        << " ,input_addr:" << input_addr << " size=" << output_size * sizeof(T);
    }
  } else {
    size_t ndim = input_storage_info->shape.size();
    int64_t storage_offset = SizeToLong(input_storage_info->storage_offset);
    auto strides = input_storage_info->strides;
    auto task = [&](size_t start, size_t end) {
      for (size_t pos = start; pos < end; ++pos) {
        int64_t tmp_idx = SizeToLong(pos);
        int64_t offset = 0;
        for (size_t dim = 0; dim < ndim; dim++) {
          auto index = ndim - dim - 1;
          int64_t mod = tmp_idx % output_shape[index];
          tmp_idx = tmp_idx / output_shape[index];
          offset += mod * strides[index];
        }
        output_addr[pos] = input_addr[offset + storage_offset];
      }
    };
    ParallelLaunch(task, output_size, 0, this, pool_);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
