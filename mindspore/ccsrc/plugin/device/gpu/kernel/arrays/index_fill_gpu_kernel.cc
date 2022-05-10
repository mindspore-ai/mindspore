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

#include "plugin/device/gpu/kernel/arrays/index_fill_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndexFillInputsNum = 4;
constexpr size_t kIndexFillOutputsNum = 1;
using kIndexType = int;
}  // namespace

bool IndexFillGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIndexFillInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIndexFillOutputsNum, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

void IndexFillGpuKernelMod::ResetResource() noexcept {
  x_shape_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

int IndexFillGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  ResetResource();
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != KRET_OK) {
    return ret;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIndexFillInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIndexFillOutputsNum, kernel_name_);
  x_shape_ = inputs.at(kIndex0)->GetShapeVector();
  x_num_ = std::accumulate(x_shape_.begin(), x_shape_.end(), 1, std::multiplies{});

  auto index_shape = inputs.at(kIndex2)->GetShapeVector();
  index_num_ = std::accumulate(index_shape.begin(), index_shape.end(), 1, std::multiplies{});
  workspace_size_list_.push_back(sizeof(bool));  // Place out_bound.
  return ret;
}

template <typename DataType, typename DimType>
bool IndexFillGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (x_num_ == 0) {
    return true;
  }
  auto x_ptr = GetDeviceAddress<DataType>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(x_ptr);
  auto dim_ptr = GetDeviceAddress<DimType>(inputs, kIndex1);
  MS_EXCEPTION_IF_NULL(dim_ptr);
  auto index_ptr = GetDeviceAddress<kIndexType>(inputs, kIndex2);
  MS_EXCEPTION_IF_NULL(index_ptr);
  auto value_ptr = GetDeviceAddress<DataType>(inputs, kIndex3);
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto y_ptr = GetDeviceAddress<DataType>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(y_ptr);
  auto out_bound_ptr = GetDeviceAddress<bool>(workspace, kIndex0);
  MS_EXCEPTION_IF_NULL(out_bound_ptr);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  // Copy from 'x' into 'y'.
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(y_ptr, x_ptr, x_num_ * sizeof(DataType), cudaMemcpyDeviceToDevice, cuda_stream),
    "cudaMemcpyAsync output 'y' from 'x' failed.");
  if (index_num_ == 0) {
    return true;
  }
  // Initialize out_bound_ptr.
  bool out_bound = false;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(out_bound_ptr, &out_bound, sizeof(bool), cudaMemcpyHostToDevice, cuda_stream),
    "cudaMemcpyAsync out_bound variable failed.");
  // Initialize and check 'dim'.
  DimType dim, rank;
  dim = rank = static_cast<DimType>(x_shape_.size());
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpy(&dim, dim_ptr, sizeof(DimType), cudaMemcpyDeviceToHost),
                                     "cudaMemcpy input 'dim' device to host failed.");
  if (dim < -rank || dim >= rank) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'dim' must be in the range [-" << rank << "," << rank
                  << "), but got " << dim;
    return false;
  } else if (dim < 0) {
    dim = dim + rank;
  }
  // Prepare index_num, dim_size, outer_size, inner_size
  int64_t dim_size = 1;
  int64_t outer_size = 1;
  int64_t inner_size = 1;
  for (size_t i = 0; i < x_shape_.size(); i++) {
    auto idx = static_cast<DimType>(i);
    if (idx < dim) {
      outer_size *= x_shape_.at(i);
    } else if (idx > dim) {
      inner_size *= x_shape_.at(i);
    } else {
      dim_size = x_shape_.at(i);
    }
  }

  int64_t real_index_num = index_num_ * (outer_size * inner_size);
  IndexFill(y_ptr, index_ptr, real_index_num, outer_size, dim_size, inner_size, value_ptr, out_bound_ptr, cuda_stream);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&out_bound, out_bound_ptr, sizeof(bool), cudaMemcpyDeviceToHost, cuda_stream),
    "cudaMemcpyAsync out_bound_ variable failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream), "IndexFill cudaStreamSynchronized failed");
  if (out_bound) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input 'index' is out of bound.";
    return false;
  }
  return true;
}

std::vector<std::pair<KernelAttr, IndexFillGpuKernelMod::IndexFillLaunchFunc>> IndexFillGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &IndexFillGpuKernelMod::LaunchKernel<uint8_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &IndexFillGpuKernelMod::LaunchKernel<uint16_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &IndexFillGpuKernelMod::LaunchKernel<uint32_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &IndexFillGpuKernelMod::LaunchKernel<uint64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &IndexFillGpuKernelMod::LaunchKernel<int8_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &IndexFillGpuKernelMod::LaunchKernel<int16_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &IndexFillGpuKernelMod::LaunchKernel<int32_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &IndexFillGpuKernelMod::LaunchKernel<int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &IndexFillGpuKernelMod::LaunchKernel<half, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &IndexFillGpuKernelMod::LaunchKernel<float, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &IndexFillGpuKernelMod::LaunchKernel<double, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &IndexFillGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &IndexFillGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &IndexFillGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &IndexFillGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &IndexFillGpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &IndexFillGpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &IndexFillGpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &IndexFillGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &IndexFillGpuKernelMod::LaunchKernel<half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &IndexFillGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &IndexFillGpuKernelMod::LaunchKernel<double, int64_t>},
};

std::vector<KernelAttr> IndexFillGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, IndexFillGpuKernelMod::IndexFillLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, IndexFill, IndexFillGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
