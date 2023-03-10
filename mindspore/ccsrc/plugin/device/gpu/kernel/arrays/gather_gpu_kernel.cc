/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <string>
#include <algorithm>

#include "plugin/device/gpu/kernel/arrays/gather_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
bool GatherFwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);

  auto input_addr = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto index_addr = reinterpret_cast<S *>(inputs.at(kIndex2)->addr);
  auto output_addr = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (!is_get_dim_) {
    auto dim_value = GetDimValue<int64_t>(inputs, kIndex1, kernel_name_, dim_type_);
    if (!SetDimParam(dim_value)) {
      return false;
    }
  }
  auto status = Gather(input_addr, index_addr, output_addr, dims_[0], dims_[1], dims_[kIndex2], dims_[kIndex3],
                       cuda_stream, GET_CTX_DEVICE_ID);
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, GatherFwdGpuKernelMod::GatherFwdFunc>> GatherFwdGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeComplex64),
   &GatherFwdGpuKernelMod::LaunchKernel<Complex<float>, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &GatherFwdGpuKernelMod::LaunchKernel<Complex<float>, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeComplex128),
   &GatherFwdGpuKernelMod::LaunchKernel<Complex<double>, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &GatherFwdGpuKernelMod::LaunchKernel<Complex<double>, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64),
   &GatherFwdGpuKernelMod::LaunchKernel<double, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &GatherFwdGpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   &GatherFwdGpuKernelMod::LaunchKernel<float, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &GatherFwdGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16),
   &GatherFwdGpuKernelMod::LaunchKernel<half, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   &GatherFwdGpuKernelMod::LaunchKernel<half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &GatherFwdGpuKernelMod::LaunchKernel<int, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   &GatherFwdGpuKernelMod::LaunchKernel<int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt8),
   &GatherFwdGpuKernelMod::LaunchKernel<int8_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8),
   &GatherFwdGpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt16),
   &GatherFwdGpuKernelMod::LaunchKernel<int16_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16),
   &GatherFwdGpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64),
   &GatherFwdGpuKernelMod::LaunchKernel<int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &GatherFwdGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &GatherFwdGpuKernelMod::LaunchKernel<uint, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt32),
   &GatherFwdGpuKernelMod::LaunchKernel<uint, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeUInt8),
   &GatherFwdGpuKernelMod::LaunchKernel<uchar, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8),
   &GatherFwdGpuKernelMod::LaunchKernel<uchar, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeBool),
   &GatherFwdGpuKernelMod::LaunchKernel<bool, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeBool),
   &GatherFwdGpuKernelMod::LaunchKernel<bool, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &GatherFwdGpuKernelMod::LaunchKernel<uint32_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt32),
   &GatherFwdGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeUInt64),
   &GatherFwdGpuKernelMod::LaunchKernel<uint64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &GatherFwdGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeUInt16),
   &GatherFwdGpuKernelMod::LaunchKernel<uint16_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt16),
   &GatherFwdGpuKernelMod::LaunchKernel<uint16_t, int64_t>}};

bool GatherFwdGpuKernelMod::SetDimParam(int64_t dim_value) {
  int64_t x_rank = SizeToLong(input_shapes_.size());
  if (dim_value < -x_rank || dim_value >= x_rank) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'dim' must be in the range [-" << x_rank << "," << x_rank
                  << "), but got " << dim_value;
    return false;
  }
  if (dim_value < 0) {
    dim_value += x_rank;
  }

  if (input_shapes_.size() <= LongToSize(dim_value) || output_shapes_.size() <= LongToSize(dim_value)) {
    MS_LOG(EXCEPTION) << "Rank should be great than " << dim_value << ", but got inputs size " << input_shapes_.size()
                      << ", outputs size: " << output_shapes_.size();
  }

  size_t dim_before_axis = 1;
  for (size_t i = 0; i < LongToSize(dim_value); i++) {
    dim_before_axis *= output_shapes_[i];
  }
  size_t dim_at_axis_input = input_shapes_[LongToSize(dim_value)];
  size_t dim_at_axis_output = output_shapes_[LongToSize(dim_value)];
  size_t dim_after_axis = 1;
  for (size_t i = LongToSize(dim_value) + 1; i < output_shapes_.size(); i++) {
    dim_after_axis *= output_shapes_[i];
  }

  dims_[0] = dim_before_axis;
  dims_[1] = dim_at_axis_input;
  dims_[kIndex2] = dim_at_axis_output;
  dims_[kIndex3] = dim_after_axis;
  return true;
}

bool GatherFwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  size_t input_num = inputs.size();
  const size_t kInputNum = 3;
  if (input_num != kInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << input_num;
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  dim_type_ = inputs.at(1)->GetDtype();
  kernel_func_ = func_list_[index].second;

  return true;
}

int GatherFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  auto input_shapes = inputs[0]->GetShapeVector();
  auto index_shapes = inputs[kIndex2]->GetShapeVector();
  auto output_shapes = outputs[0]->GetShapeVector();

  input_shapes_.clear();
  index_shapes_.clear();
  output_shapes_.clear();
  std::transform(input_shapes.cbegin(), input_shapes.cend(), std::back_inserter(input_shapes_), LongToSize);
  std::transform(index_shapes.cbegin(), index_shapes.cend(), std::back_inserter(index_shapes_), LongToSize);
  std::transform(output_shapes.cbegin(), output_shapes.cend(), std::back_inserter(output_shapes_), LongToSize);

  is_null_input_ = CHECK_SHAPE_NULL(input_shapes_, kernel_name_, "input") ||
                   CHECK_SHAPE_NULL(index_shapes_, kernel_name_, "input_indices") ||
                   CHECK_SHAPE_NULL(output_shapes_, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_OK;
  }

  if (input_shapes_.size() != index_shapes_.size() || input_shapes_.size() != output_shapes_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input and output must be the equal to the "
                  << "dimension of index: " << index_shapes_.size()
                  << ", but got the dimension of input: " << input_shapes_.size()
                  << ", the dimension of output: " << output_shapes_.size();
    return KRET_RESIZE_FAILED;
  }

  int64_t dim_value = 0;

  if (TryGetIntValue(inputs, 1, kernel_name_, &dim_value)) {
    is_get_dim_ = true;
    if (!SetDimParam(dim_value)) {
      return KRET_RESIZE_FAILED;
    }
  }

  return KRET_OK;
}

std::vector<KernelAttr> GatherFwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GatherFwdFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, GatherD, GatherFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
