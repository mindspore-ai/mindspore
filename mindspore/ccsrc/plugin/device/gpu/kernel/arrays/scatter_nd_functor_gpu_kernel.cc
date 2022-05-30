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

#include "plugin/device/gpu/kernel/arrays/scatter_nd_functor_gpu_kernel.h"
#include <type_traits>
#include <numeric>
#include <functional>
#include <algorithm>
#include <utility>
#include <memory>
#include <map>
#include "abstract/utils.h"

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
bool ScatterNdFunctorGPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices = GetDeviceAddress<S>(inputs, kIndex1);
  T *updates = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  const size_t indices_len = sizeof(S) * out_strides_.size();
  S *indices_stride = GetDeviceAddress<S>(workspace, kIndex0);

  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr_);

  // The out_strides_ used to be std::vector<S>, use int as default outside
  if constexpr (std::is_same_v<S, int64_t>) {
    std::vector<int64_t> long_out_stride;
    std::transform(out_strides_.begin(), out_strides_.end(), std::back_inserter(long_out_stride), IntToLong);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(indices_stride, long_out_stride.data(), indices_len, cudaMemcpyHostToDevice, cuda_stream),
      "For 'ScatterNdFunctorGPUKernelMod', cudaMemcpyAsync failed in ScatterNdFunctorGpuFwdKernel::LaunchKernel.")
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(indices_stride, out_strides_.data(), indices_len, cudaMemcpyHostToDevice, cuda_stream),
      "For 'ScatterNdFunctorGPUKernelMod', cudaMemcpyAsync failed in ScatterNdFunctorGpuFwdKernel::LaunchKernel.")
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&output[0], &input[0], input_size_ * sizeof(T), cudaMemcpyDeviceToDevice, cuda_stream),
    "For 'ScatterNdFunctorGPUKernelMod', cudaMemcpyAsync output failed")

  CalScatterNdFunctor(scatter_nd_functor_type_, unit_size_, num_units_, index_depth_, indices_stride, indices, updates,
                      output, device_id_, cuda_stream);
  return true;
}

bool ScatterNdFunctorGPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto iter = kScatterNdFunctorTypeMap.find(kernel_name_);
  if (iter == kScatterNdFunctorTypeMap.end()) {
    MS_LOG(EXCEPTION) << "Only support these scatter functors: " << Map2Str(kScatterNdFunctorTypeMap)
                      << " currently, but got " << kernel_name_;
  }
  scatter_nd_functor_type_ = iter->second;

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int ScatterNdFunctorGPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  // Shape in new API is std::vector<int64_t>, need to adapt to old api
  const auto input_shape_64 = inputs.at(kIndex0)->GetShapeVector();
  const auto indices_shape_64 = inputs.at(kIndex1)->GetShapeVector();
  const auto updates_shape_64 = inputs.at(kIndex2)->GetShapeVector();
  std::vector<size_t> input_shape, indices_shape, updates_shape;
  (void)std::transform(input_shape_64.begin(), input_shape_64.end(), std::back_inserter(input_shape), LongToSize);
  (void)std::transform(indices_shape_64.begin(), indices_shape_64.end(), std::back_inserter(indices_shape), LongToSize);
  (void)std::transform(updates_shape_64.begin(), updates_shape_64.end(), std::back_inserter(updates_shape), LongToSize);
  auto index_depth = indices_shape.back();

  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") ||
                   CHECK_SHAPE_NULL(indices_shape, kernel_name_, "indices") ||
                   CHECK_SHAPE_NULL(updates_shape, kernel_name_, "updates");

  if (indices_shape.size() < kMinIndiceRank) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the dimension of 'indices' must be at least 2, but got "
                             << indices_shape.size();
  }

  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies{});

  unit_size_ = 1;
  for (size_t i = indices_shape.size() - 1; i < updates_shape.size(); ++i) {
    unit_size_ *= SizeToInt(updates_shape[i]);
  }

  num_units_ = 1;
  num_units_ *= updates_shape[indices_shape.size() - 2];
  for (int i = SizeToInt(indices_shape.size()) - 3; i >= 0; i--) {
    num_units_ *= updates_shape[i];
  }

  index_depth_ = SizeToInt(index_depth);
  int32_t out_stride = 1;
  out_strides_.push_back(out_stride);
  for (int i = SizeToInt(index_depth_) - 2; i >= 0; i--) {
    out_stride *= SizeToInt(input_shape[i + 1]);
    out_strides_.push_back(out_stride);
  }
  reverse(out_strides_.begin(), out_strides_.end());

  const auto index_size = abstract::TypeIdSize(inputs.at(kIndex1)->GetDtype());
  workspace_size_list_ = {out_strides_.size() * index_size};

  return KRET_OK;
}

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
    DTYPE_REGISTER(kNumberTypeUInt64, kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, uint64_t, int),
    DTYPE_REGISTER(kNumberTypeUInt64, kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, uint64_t, int64_t),
    // Data type: int
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int, int),
    DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int, int64_t),
    // Data type: int16_t
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt16, int16_t, int),
    DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt16, int16_t, int64_t),
    // Data type: uint8_t
    DTYPE_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t, int),
    DTYPE_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uint8_t, int64_t),
    // Data type: int8_t
    DTYPE_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, int8_t, int),
    DTYPE_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, int8_t, int64_t),
    // Data type: bool, only for scatter_nd_update
    DTYPE_REGISTER(kNumberTypeBool, kNumberTypeInt32, kNumberTypeBool, kNumberTypeBool, bool, int),
    DTYPE_REGISTER(kNumberTypeBool, kNumberTypeInt64, kNumberTypeBool, kNumberTypeBool, bool, int64_t),
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
