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

#include "plugin/device/gpu/kernel/sparse/sparse_tensor_dense_add_gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kSparseTensorDenseAddInputsNum = 4;
constexpr int kSparseTensorDenseAddOutputsNum = 1;
constexpr int kSparseTensorDenseAddIndex0 = 0;
constexpr int kSparseTensorDenseAddIndex3 = 3;
}  // namespace

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, SparseTensorDenseAddGpuKernelMod::SparseTensorDenseAddLaunchFunc>>
  SparseTensorDenseAddGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeUInt8),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeUInt8),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt8),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<int8_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt8),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeUInt16),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeUInt16),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<Complex<float>, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<Complex<float>, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<Complex<double>, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SparseTensorDenseAddGpuKernelMod::LaunchKernel<Complex<double>, int64_t>},
};

template <typename T, typename I>
bool SparseTensorDenseAddGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> &workspace,
                                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseTensorDenseAddInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseTensorDenseAddOutputsNum, kernel_name_);
  I *x1_indices_addr = GetDeviceAddress<I>(inputs, 0);
  T *x1_values_addr = GetDeviceAddress<T>(inputs, 1);
  I *x1_shape_addr = GetDeviceAddress<I>(inputs, 2);
  T *x2_values_addr = GetDeviceAddress<T>(inputs, 3);
  T *y_addr = GetDeviceAddress<T>(outputs, 0);

  size_t *x2_shape = GetDeviceAddress<size_t>(workspace, 0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(x2_shape, &x2_shape_[0], workspace_size_, cudaMemcpyHostToDevice,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "cudaMemcpyAsync x2_shape failed");
  constexpr int X1_SHAPE_INDICES = 2;
  std::vector<I> x1_shape(inputs[X1_SHAPE_INDICES]->size / sizeof(I));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(x1_shape.data(), x1_shape_addr, inputs[X1_SHAPE_INDICES]->size, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cudaMemcpyAsync x1_shape failed");

  std::vector<I> x1_indices_host(inputs[0]->size / sizeof(I));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(x1_indices_host.data(), x1_indices_addr, inputs[0]->size, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cudaMemcpyAsync x1_indices failed");

  if (x1_shape.size() != x2_shape_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << " The input x1_shape size does not equal x2_shape size! "
                  << "tensor shape of 'sparse': " << x1_shape.size()
                  << ",and the tensor shape of 'dense':" << x2_shape_.size();
    return false;
  }

  for (size_t idx = 0; idx < x2_shape_.size(); ++idx) {
    if (x1_shape[idx] != x2_shape_[idx]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << " The input x1_shape dim does not equal x2_shape dim! "
                    << "tensor dim of 'sparse': " << x1_shape[idx]
                    << ",and the tensor dim of 'dense':" << x2_shape_[idx];
      return false;
    }
    if (x1_indices_host[idx] >= x1_shape[idx]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << " The input x1_indices is out of bounds! "
                    << "x1_indices is : " << x1_indices_host[idx] << ", tensor bounds is:" << x1_shape[idx];
      return false;
    }
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(y_addr, x2_values_addr, output_elements_ * sizeof(T), cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "Cuda memcopy input to output Fail");
  SparseTensorDenseAddKernel(static_cast<size_t>(input_elements_), static_cast<size_t>(rank_), x2_shape,
                             x1_indices_addr, x1_values_addr, x1_shape_addr, x2_values_addr, y_addr, device_id_,
                             reinterpret_cast<cudaStream_t>(cuda_stream_));

  return true;
}

bool SparseTensorDenseAddGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  if (inputs.size() != kSparseTensorDenseAddInputsNum || outputs.size() != kSparseTensorDenseAddOutputsNum) {
    MS_LOG(ERROR) << "For 'SparseTensorDenseAdd', input and output size must be " << kSparseTensorDenseAddInputsNum
                  << " and " << kSparseTensorDenseAddOutputsNum << ", but got " << inputs.size() << " and "
                  << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'SparseTensorDenseAdd' does not support this kernel data type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[pair.second].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  return true;
}

int SparseTensorDenseAddGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    MS_LOG(ERROR) << kernel_name_ << " reinit failed.";
    return ret;
  }
  std::vector<int64_t> input_shape_0 = inputs[kSparseTensorDenseAddIndex0]->GetShapeVector();
  x2_shape_ = inputs[kSparseTensorDenseAddIndex3]->GetShapeVector();
  std::vector<int64_t> output_shape_ = outputs[kSparseTensorDenseAddIndex0]->GetShapeVector();
  is_null_output_ = CHECK_SHAPE_NULL(x2_shape_, kernel_name_, "x2_input");
  if (is_null_output_) {
    InitSizeLists();
    return true;
  }
  x2_shape_size = x2_shape_.size();
  auto output_shape_size = output_shape_.size();
  if (x2_shape_size != output_shape_size) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input four shape size should be the same as"
                  << " output shape size, but got input four shape size " << x2_shape_ << " output shape size"
                  << output_shape_size;
    return KRET_RESIZE_FAILED;
  }
  // A Code Block For setting input and output shape.
  input_elements_ = input_shape_0[0];
  output_elements_ = 1;
  for (size_t i = 0; i < output_shape_size; i++) {
    output_elements_ *= output_shape_[i];
  }
  rank_ = x2_shape_.size();
  InitSizeLists();
  return KRET_OK;
}

std::vector<KernelAttr> SparseTensorDenseAddGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseTensorDenseAddLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseTensorDenseAdd, SparseTensorDenseAddGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
