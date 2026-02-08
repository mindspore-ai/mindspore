/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/fill_gpu_kernel.h"
#include <limits>
#include <memory>
#include <functional>
#include <algorithm>
#include "base/float16.h"
#include "mindspore/ops/infer/fill.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/fill_impl.cuh"
#include "include/utils/utils.h"

namespace mindspore {
namespace kernel {
#define FILL_GPU_REG(MS_T, MS_U, MS_V, T) \
  {KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_U).AddOutputAttr(MS_V), &FillGpuKernelMod::LaunchKernel<T>}

template <typename T>
T FillGpuKernelMod::GetInputDataFromDevice(const std::vector<KernelTensor *> &inputs, size_t idx,
                                           cudaStream_t cuda_stream) {
  auto value_ptr = GetDeviceAddress<T>(inputs, idx);
  T original_value;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&original_value, value_ptr, sizeof(T), cudaMemcpyDeviceToHost, cuda_stream),
    "For 'Fill', cudaMemcpyAsync value variable failed.");
  if (cudaStreamQuery(cuda_stream) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream), "cuda Stream Sync Failed.");
  }
  return original_value;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
const std::vector<std::pair<KernelAttr, FillGpuKernelMod::KernelRunFunc>> &FillGpuKernelMod::GetFuncList() const {
  static std::vector<std::pair<KernelAttr, FillGpuKernelMod::KernelRunFunc>> func_list;
  std::vector<TypeId> shape_type_list = {kNumberTypeInt32, kNumberTypeInt64};
  std::vector<TypeId> value_type_list = {
    kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,     kNumberTypeInt64,     kNumberTypeFloat16,
    kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeUInt8,     kNumberTypeUInt16,    kNumberTypeUInt32,
    kNumberTypeUInt64,  kNumberTypeBool,    kNumberTypeComplex64, kNumberTypeComplex128};

  if (func_list.empty()) {
    std::pair<KernelAttr, FillGpuKernelMod::KernelRunFunc> type_pair;
    for (auto i : shape_type_list) {
      for (auto j : value_type_list) {
        for (auto k : value_type_list) {
          if (k == kNumberTypeInt8) {
            type_pair = FILL_GPU_REG(i, j, k, int8_t);
          } else if (k == kNumberTypeInt16) {
            type_pair = FILL_GPU_REG(i, j, k, int16_t);
          } else if (k == kNumberTypeInt32) {
            type_pair = FILL_GPU_REG(i, j, k, int32_t);
          } else if (k == kNumberTypeInt64) {
            type_pair = FILL_GPU_REG(i, j, k, int64_t);
          } else if (k == kNumberTypeFloat16) {
            type_pair = FILL_GPU_REG(i, j, k, float16);
          } else if (k == kNumberTypeFloat32) {
            type_pair = FILL_GPU_REG(i, j, k, float);
          } else if (k == kNumberTypeFloat64) {
            type_pair = FILL_GPU_REG(i, j, k, double);
          } else if (k == kNumberTypeUInt8) {
            type_pair = FILL_GPU_REG(i, j, k, uint8_t);
          } else if (k == kNumberTypeUInt16) {
            type_pair = FILL_GPU_REG(i, j, k, uint16_t);
          } else if (k == kNumberTypeUInt32) {
            type_pair = FILL_GPU_REG(i, j, k, uint32_t);
          } else if (k == kNumberTypeUInt64) {
            type_pair = FILL_GPU_REG(i, j, k, uint64_t);
          } else if (k == kNumberTypeBool) {
            type_pair = FILL_GPU_REG(i, j, k, bool);
          } else if (k == kNumberTypeComplex64) {
            type_pair = FILL_GPU_REG(i, j, k, Complex<float>);
          } else if (k == kNumberTypeComplex128) {
            type_pair = FILL_GPU_REG(i, j, k, Complex<double>);
          }
          func_list.emplace_back(type_pair);
        }
      }
    }
  }
  return func_list;
}

bool FillGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  x_type_id_ = tensor_attr.GetInputAttr(kIndex1).dtype;
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int FillGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(inputs, outputs)) != KRET_OK) {
    return ret;
  }
  auto shape = outputs.at(kIndex0)->GetShapeVector();
  input_elements_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  auto workspace_size = sizeof(double);
  workspace_size_list_.emplace_back(workspace_size);

  return KRET_OK;
}

template <typename T>
bool FillGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs) {
  T value;
  auto cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  if (x_type_id_ == kNumberTypeInt8) {
    value = static_cast<T>(GetInputDataFromDevice<int8_t>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeInt16) {
    value = static_cast<T>(GetInputDataFromDevice<int16_t>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeInt32) {
    value = static_cast<T>(GetInputDataFromDevice<int32_t>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeInt64) {
    value = static_cast<T>(GetInputDataFromDevice<int64_t>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeUInt8) {
    value = static_cast<T>(GetInputDataFromDevice<uint8_t>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeUInt16) {
    value = static_cast<T>(GetInputDataFromDevice<uint16_t>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeUInt32) {
    value = static_cast<T>(GetInputDataFromDevice<uint32_t>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeUInt64) {
    value = static_cast<T>(GetInputDataFromDevice<uint64_t>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeFloat16) {
    value = static_cast<T>(GetInputDataFromDevice<float16>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeFloat32) {
    value = static_cast<T>(GetInputDataFromDevice<float>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeFloat64) {
    value = static_cast<T>(GetInputDataFromDevice<double>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeBool) {
    value = static_cast<T>(GetInputDataFromDevice<bool>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeComplex64) {
    value = static_cast<T>(GetInputDataFromDevice<Complex<float>>(inputs, kIndex1, cuda_stream));
  } else if (x_type_id_ == kNumberTypeComplex128) {
    value = static_cast<T>(GetInputDataFromDevice<Complex<double>>(inputs, kIndex1, cuda_stream));
  }
  auto y_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(y_ptr, &value, sizeof(T), cudaMemcpyHostToDevice, cuda_stream),
                                     "cudaMemcpy value variable failed.");
  cudaError_t status = cudaErrorNotReady;
  if (std::is_same<T, float16>::value) {
    status = Fill(input_elements_, 1, reinterpret_cast<half *>(y_ptr), reinterpret_cast<half *>(y_ptr), cuda_stream);
  } else {
    status = Fill(input_elements_, 1, y_ptr, y_ptr, cuda_stream);
  }
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Fill, FillGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
