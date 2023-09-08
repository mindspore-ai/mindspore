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
#include <complex>
#include <functional>
#include "utils/log_adapter.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

std::unordered_map<TypeId, ContiguousCpuKernel::ContiguousFunc> ContiguousCpuKernel::func_list_ = {
  {kNumberTypeFloat16, &ContiguousCpuKernel::LaunchContiguousImpl<float16>},
  {kNumberTypeFloat32, &ContiguousCpuKernel::LaunchContiguousImpl<float>},
  {kNumberTypeFloat64, &ContiguousCpuKernel::LaunchContiguousImpl<double>},
  {kNumberTypeInt8, &ContiguousCpuKernel::LaunchContiguousImpl<int8_t>},
  {kNumberTypeInt16, &ContiguousCpuKernel::LaunchContiguousImpl<int16_t>},
  {kNumberTypeInt32, &ContiguousCpuKernel::LaunchContiguousImpl<int32_t>},
  {kNumberTypeInt64, &ContiguousCpuKernel::LaunchContiguousImpl<int64_t>},
  {kNumberTypeBool, &ContiguousCpuKernel::LaunchContiguousImpl<bool>},
  {kNumberTypeComplex64, &ContiguousCpuKernel::LaunchContiguousImpl<complex64>},
  {kNumberTypeComplex128, &ContiguousCpuKernel::LaunchContiguousImpl<complex128>},
  {kNumberTypeUInt8, &ContiguousCpuKernel::LaunchContiguousImpl<uint8_t>},
  {kNumberTypeUInt16, &ContiguousCpuKernel::LaunchContiguousImpl<uint16_t>},
  {kNumberTypeUInt32, &ContiguousCpuKernel::LaunchContiguousImpl<uint32_t>},
  {kNumberTypeUInt64, &ContiguousCpuKernel::LaunchContiguousImpl<uint64_t>}};

bool ContiguousCpuKernel::LaunchContiguous(TypeId type_id, const kernel::AddressPtr &input,
                                           const TensorStorageInfoPtr &input_storage_info,
                                           const kernel::AddressPtr &output) {
  const auto &iter = func_list_.find(type_id);
  if (iter == func_list_.end()) {
    MS_LOG(EXCEPTION) << "type_id:" << type_id << " is invalid";
  }

  return iter->second(this, input, input_storage_info, output);
}

template <typename T>
bool ContiguousCpuKernel::LaunchContiguousImpl(const kernel::AddressPtr &input,
                                               const TensorStorageInfoPtr &input_storage_info,
                                               const kernel::AddressPtr &output) {
  MS_EXCEPTION_IF_NULL(input_storage_info);
  T *input_addr = GetDeviceAddress<T>({input}, 0);
  T *output_addr = GetDeviceAddress<T>({output}, 0);
  const auto &output_shape = input_storage_info->shape;
  auto output_size =
    LongToSize(std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>()));
  if (output_size == 0) {
    // CPU unsupported zero copy
    return true;
  }
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
    size_t storage_offset = input_storage_info->storage_offset;
    auto strides = input_storage_info->strides;
    auto task = [&](size_t start, size_t end) {
      for (size_t pos = start; pos < end; ++pos) {
        int64_t tmp_idx = pos;
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
