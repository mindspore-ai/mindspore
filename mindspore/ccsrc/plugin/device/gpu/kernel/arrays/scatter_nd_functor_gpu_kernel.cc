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
#include <map>
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
static const std::map<std::string, ScatterNdFunctorType> kScatterNdFunctorTypeMap = {
  {"ScatterNdUpdate", SCATTER_ND_FUNC_UPDATE}, {"ScatterNdAdd", SCATTER_ND_FUNC_ADD},
  {"ScatterNdSub", SCATTER_ND_FUNC_SUB},       {"ScatterNdMul", SCATTER_ND_FUNC_MUL},
  {"ScatterNdDiv", SCATTER_ND_FUNC_DIV},       {"ScatterNdMax", SCATTER_ND_FUNC_MAX},
  {"ScatterNdMin", SCATTER_ND_FUNC_MIN},
};
}
template <typename T, typename S>
bool ScatterNdFunctorKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices = GetDeviceAddress<S>(inputs, kIndex1);
  T *updates = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  const size_t indices_len = sizeof(S) * out_strides_.size();
  S *indices_stride = GetDeviceAddress<S>(workspace, 0);

  // The out_strides_ used to be std::vector<S>, use int as default outside
  if constexpr (std::is_same_v<S, int64_t>) {
    std::vector<int64_t> long_out_stride;
    std::transform(out_strides_.begin(), out_strides_.end(), std::back_inserter(long_out_stride), IntToLong);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(indices_stride, long_out_stride.data(), indices_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For '" << kernel_name_ << "', cudaMemcpyAsync failed in ScatterNdFunctorGpuFwdKernel::LaunchKernel.")
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(indices_stride, out_strides_.data(), indices_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For '" << kernel_name_ << "', cudaMemcpyAsync failed in ScatterNdFunctorGpuFwdKernel::LaunchKernel.")
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&output[0], &input[0], input_size_ * sizeof(T), cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "For '" << kernel_name_ << "', cudaMemcpyAsync output failed")
  CalScatterNdFunctor(scatter_nd_functor_type_, unit_size_, num_units_, index_depth_, indices_stride, indices, updates,
                      output, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

bool ScatterNdFunctorKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto iter = kScatterNdFunctorTypeMap.find(kernel_name_);
  if (iter == kScatterNdFunctorTypeMap.end()) {
    MS_LOG(EXCEPTION) << "Only support these scatter functors: " << Map2Str(kScatterNdFunctorTypeMap)
                      << " currently, but got " << kernel_name_;
  } else {
    scatter_nd_functor_type_ = iter->second;
  }

  // Getting launch_kernel function.
  {
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    // Only ScatterNdUpdate support kNumberTypeBool
    if (scatter_nd_functor_type_ != SCATTER_ND_FUNC_UPDATE &&
        kernel_attr.GetInputAttr(kIndex0).first == kNumberTypeBool) {
      is_match = false;
    }
    if (!is_match) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
      return false;
    }
    kernel_func_ = func_list_[index].second;
  }

  return true;
}
int ScatterNdFunctorKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

std::vector<KernelAttr> ScatterNdFunctorKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ScatterNdFunctorFunc> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, ScatterNdFunctorKernelMod::ScatterNdFunctorFunc>>
  ScatterNdFunctorKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &ScatterNdFunctorKernelMod::LaunchKernel<double, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &ScatterNdFunctorKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ScatterNdFunctorKernelMod::LaunchKernel<float, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ScatterNdFunctorKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ScatterNdFunctorKernelMod::LaunchKernel<half, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ScatterNdFunctorKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &ScatterNdFunctorKernelMod::LaunchKernel<int, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &ScatterNdFunctorKernelMod::LaunchKernel<int, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16),
     &ScatterNdFunctorKernelMod::LaunchKernel<int16_t, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16),
     &ScatterNdFunctorKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeUInt8),
     &ScatterNdFunctorKernelMod::LaunchKernel<uint8_t, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeUInt8),
     &ScatterNdFunctorKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt8),
     &ScatterNdFunctorKernelMod::LaunchKernel<int8_t, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt8),
     &ScatterNdFunctorKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeBool),
     &ScatterNdFunctorKernelMod::LaunchKernel<bool, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeBool),
     &ScatterNdFunctorKernelMod::LaunchKernel<bool, int64_t>},
};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdUpdate, ScatterNdFunctorKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdAdd, ScatterNdFunctorKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdSub, ScatterNdFunctorKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdMul, ScatterNdFunctorKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdDiv, ScatterNdFunctorKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdMax, ScatterNdFunctorKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ScatterNdMin, ScatterNdFunctorKernelMod);
}  // namespace kernel
}  // namespace mindspore
