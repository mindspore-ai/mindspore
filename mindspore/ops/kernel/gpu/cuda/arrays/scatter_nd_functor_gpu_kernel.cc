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

#include "kernel/gpu/cuda/arrays/scatter_nd_functor_gpu_kernel.h"
#include <complex>
#include <type_traits>
#include <numeric>
#include <functional>
#include <algorithm>
#include <utility>
#include <memory>
#include <map>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMinIndiceRank = 2;
static const std::map<std::string, ScatterNdFunctorType> kScatterNdFunctorTypeMap = {
  {"ScatterNdUpdate", SCATTER_ND_FUNC_UPDATE}, {"ScatterNdAdd", SCATTER_ND_FUNC_ADD},
  {"ScatterNdSub", SCATTER_ND_FUNC_SUB},       {"ScatterNdMul", SCATTER_ND_FUNC_MUL},
  {"ScatterNdDiv", SCATTER_ND_FUNC_DIV},       {"ScatterNdMax", SCATTER_ND_FUNC_MAX},
  {"ScatterNdMin", SCATTER_ND_FUNC_MIN},
};
}  // namespace
using KernelRunFunc = ScatterNdFunctorGPUKernelMod::KernelRunFunc;
template <typename T, typename S>
bool ScatterNdFunctorGPUKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &workspace,
                                                const std::vector<KernelTensor *> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices = GetDeviceAddress<S>(inputs, kIndex1);
  T *updates = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  const size_t indices_len = sizeof(S) * out_strides_.size();
  const size_t work_shape_len = sizeof(S) * work_shape_.size();
  S *indices_stride = GetDeviceAddress<S>(workspace, kIndex0);
  S *work_shape = GetDeviceAddress<S>(workspace, kIndex1);

  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr_);

  // The out_strides_ used to be std::vector<S>, use int as default outside
  if constexpr (std::is_same_v<S, int64_t>) {
    std::vector<int64_t> long_out_stride;
    std::vector<int64_t> long_work_shape;
    std::transform(out_strides_.begin(), out_strides_.end(), std::back_inserter(long_out_stride), IntToLong);
    std::transform(work_shape_.begin(), work_shape_.end(), std::back_inserter(long_work_shape), IntToLong);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(indices_stride, long_out_stride.data(), indices_len, cudaMemcpyHostToDevice, cuda_stream),
      "For 'ScatterNdFunctorGPUKernelMod', cudaMemcpyAsync failed in ScatterNdFunctorGpuFwdKernel::LaunchKernel.")
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(work_shape, long_work_shape.data(), work_shape_len, cudaMemcpyHostToDevice, cuda_stream),
      "For 'ScatterNdFunctorGPUKernelMod', cudaMemcpyAsync failed in ScatterNdFunctorGpuFwdKernel::LaunchKernel.")
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(indices_stride, out_strides_.data(), indices_len, cudaMemcpyHostToDevice, cuda_stream),
      "For 'ScatterNdFunctorGPUKernelMod', cudaMemcpyAsync failed in ScatterNdFunctorGpuFwdKernel::LaunchKernel.")
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(work_shape, work_shape_.data(), work_shape_len, cudaMemcpyHostToDevice, cuda_stream),
      "For 'ScatterNdFunctorGPUKernelMod', cudaMemcpyAsync failed in ScatterNdFunctorGpuFwdKernel::LaunchKernel.")
  }

  auto status = CalScatterNdFunctor(scatter_nd_functor_type_, unit_size_, num_units_, index_depth_, indices_stride,
                                    indices, work_shape, updates, input, device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output, input, inputs[0]->size(), cudaMemcpyDeviceToDevice, cuda_stream),
    "For 'ScatterNdFunctorGPUKernelMod', cudaMemcpyAsync output failed")

  return true;
}

bool ScatterNdFunctorGPUKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto iter = kScatterNdFunctorTypeMap.find(kernel_name_);
  if (iter == kScatterNdFunctorTypeMap.end()) {
    MS_LOG(EXCEPTION) << "Only support these scatter functors: "
                      << Map2Str<std::map, ScatterNdFunctorType>(kScatterNdFunctorTypeMap) << " currently, but got "
                      << kernel_name_;
  }
  scatter_nd_functor_type_ = iter->second;

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  if (scatter_nd_functor_type_ != SCATTER_ND_FUNC_UPDATE && (inputs[kIndex0]->dtype_id() == kNumberTypeBool)) {
    const auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }

  return true;
}

int ScatterNdFunctorGPUKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  // Shape in new API is std::vector<int64_t>, need to adapt to old api
  std::vector<size_t> input_shape = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  std::vector<size_t> indices_shape = LongVecToSizeVec(inputs.at(kIndex1)->GetShapeVector());
  std::vector<size_t> updates_shape = LongVecToSizeVec(inputs.at(kIndex2)->GetShapeVector());
  auto index_depth = indices_shape.back();

  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") ||
                   CHECK_SHAPE_NULL(indices_shape, kernel_name_, "indices") ||
                   CHECK_SHAPE_NULL(updates_shape, kernel_name_, "updates");
  if (is_null_input_) {
    return KRET_OK;
  }

  if (indices_shape.size() < kMinIndiceRank) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the dimension of 'indices' must be at least 2, but got "
                             << indices_shape.size();
  }

  unit_size_ = 1;
  for (size_t i = indices_shape.size() - 1; i < updates_shape.size(); ++i) {
    unit_size_ *= SizeToInt(updates_shape[i]);
  }

  num_units_ = 1;
  num_units_ *= updates_shape[indices_shape.size() - kMinIndiceRank];
  for (int i = SizeToInt(indices_shape.size()) - (kMinIndiceRank + 1); i >= 0; i--) {
    num_units_ *= updates_shape[i];
  }

  index_depth_ = SizeToInt(index_depth);
  out_strides_.clear();
  int32_t out_stride = 1;
  out_strides_.push_back(out_stride);
  for (int i = SizeToInt(index_depth_) - kMinIndiceRank; i >= 0; i--) {
    out_stride *= SizeToInt(input_shape[i + 1]);
    out_strides_.push_back(out_stride);
  }
  reverse(out_strides_.begin(), out_strides_.end());

  work_shape_.clear();
  work_shape_ = std::vector<int32_t>(input_shape.begin(), input_shape.end());

  const auto index_size = abstract::TypeIdSize(inputs.at(kIndex1)->dtype_id());
  workspace_size_list_ = {
    out_strides_.size() * index_size,
    work_shape_.size() * index_size,
  };

  return KRET_OK;
}

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

#define DTYPE_REGISTER(INPUT, INDICES, UPDATES, OUTPUT, T, S)                                           \
  {                                                                                                     \
    KernelAttr().AddInputAttr(INPUT).AddInputAttr(INDICES).AddInputAttr(UPDATES).AddOutputAttr(OUTPUT), \
      &ScatterNdFunctorGPUKernelMod::LaunchKernel<T, S>                                                 \
  }

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &ScatterNdFunctorGPUKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    // Data type: double
    DTYPE_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64, double, int),
    DTYPE_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64, double, int64_t),
    // Data type: float
    DTYPE_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32, float, int),
    DTYPE_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32, float, int64_t),
    // Data type: half
    DTYPE_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16, half, int),
    DTYPE_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16, half, int64_t),
    // Data type: int64
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int64_t, int),
    DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
    // Data type: uint64
    DTYPE_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeUInt64, uint64_t, int),
    DTYPE_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, kNumberTypeUInt64, uint64_t, int64_t),
    // Data type: int32_t
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t, int),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int32_t, int64_t),
    // Data type: uint32_t
    DTYPE_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeUInt32, kNumberTypeUInt32, uint32_t, int),
    DTYPE_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeUInt32, kNumberTypeUInt32, uint32_t, int64_t),
    // Data type: int16_t
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt16, int16_t, int),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt16, int16_t, int64_t),
    // Data type: uint16_t
    DTYPE_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeUInt16, kNumberTypeUInt16, uint16_t, int),
    DTYPE_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeUInt16, kNumberTypeUInt16, uint16_t, int64_t),
    // Data type: int8_t
    DTYPE_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, int8_t, int),
    DTYPE_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, int8_t, int64_t),
    // Data type: uint8_t
    DTYPE_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t, int),
    DTYPE_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t, int64_t),
    // Data type: bool, only for scatter_nd_update
    DTYPE_REGISTER(kNumberTypeBool, kNumberTypeInt32, kNumberTypeBool, kNumberTypeBool, bool, int),
    DTYPE_REGISTER(kNumberTypeBool, kNumberTypeInt64, kNumberTypeBool, kNumberTypeBool, bool, int64_t),
    // Data type: complex64, only for scatter_nd_update
    DTYPE_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, kNumberTypeComplex64, kNumberTypeComplex64, complex64, int),
    DTYPE_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeComplex64, kNumberTypeComplex64, complex64,
                   int64_t),
    // Data type: complex128, only for scatter_nd_update
    DTYPE_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, kNumberTypeComplex128, kNumberTypeComplex128, complex128,
                   int),
    DTYPE_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, kNumberTypeComplex128, kNumberTypeComplex128, complex128,
                   int64_t),
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdUpdate, ScatterNdFunctorGPUKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdAdd, ScatterNdFunctorGPUKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdSub, ScatterNdFunctorGPUKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdMul, ScatterNdFunctorGPUKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdDiv, ScatterNdFunctorGPUKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdMax, ScatterNdFunctorGPUKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdMin, ScatterNdFunctorGPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
