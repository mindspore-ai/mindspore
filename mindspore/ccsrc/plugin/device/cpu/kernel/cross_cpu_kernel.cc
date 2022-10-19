/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/cross_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/cross.h"

namespace {
const size_t kDataSizeThreshold = 4 * 1024;
const size_t kNumber0 = 0;
const size_t kNumber1 = 1;
const size_t kNumber3 = 3;
}  // namespace

namespace mindspore {
namespace kernel {
bool CrossCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int CrossCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input1_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input2_shape_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kIndex0]->GetDeviceShapeAdaptively();
  input1_dtype_ = inputs[kIndex0]->GetDtype();
  auto cross_ptr = std::dynamic_pointer_cast<ops::Cross>(base_operator);
  MS_EXCEPTION_IF_NULL(cross_ptr);
  dim_ = cross_ptr->get_dim();
  int64_t default_dim = -65530;
  if (dim_ == default_dim) {
    int64_t dim_size_value = 3;
    for (size_t i = 0; i < input1_shape_.size(); i++) {
      if (input1_shape_[i] == dim_size_value) {
        dim_ = static_cast<int64_t>(i);
        break;
      }
      if (i == input1_shape_.size() - 1 && input1_shape_[i] != dim_size_value) {
        MS_EXCEPTION(ValueError) << "The size of inputs dim must be 3,but got" << input1_shape_[i];
      }
    }
  }
  if (dim_ < 0) {
    dim_ = static_cast<int64_t>(input1_shape_.size()) + dim_;
  }
  return KRET_OK;
}

bool CrossCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  switch (input1_dtype_) {
    case kNumberTypeInt8:
      return LaunchKernel<int8_t>(inputs, outputs);
    case kNumberTypeInt16:
      return LaunchKernel<int16_t>(inputs, outputs);
    case kNumberTypeInt32:
      return LaunchKernel<int32_t>(inputs, outputs);
    case kNumberTypeInt64:
      return LaunchKernel<int64_t>(inputs, outputs);
    case kNumberTypeUInt8:
      return LaunchKernel<uint8_t>(inputs, outputs);
    case kNumberTypeUInt16:
      return LaunchKernel<uint16_t>(inputs, outputs);
    case kNumberTypeUInt32:
      return LaunchKernel<uint32_t>(inputs, outputs);
    case kNumberTypeUInt64:
      return LaunchKernel<uint64_t>(inputs, outputs);
    case kNumberTypeFloat16:
      return LaunchKernel<float16>(inputs, outputs);
    case kNumberTypeFloat32:
      return LaunchKernel<float>(inputs, outputs);
    case kNumberTypeFloat64:
      return LaunchKernel<double>(inputs, outputs);
    case kNumberTypeComplex64:
      return LaunchKernel<std::complex<float>>(inputs, outputs);
    case kNumberTypeComplex128:
      return LaunchKernel<std::complex<double>>(inputs, outputs);
    default:
      MS_EXCEPTION(TypeError) << "Unsupported input data type: " << input1_dtype_;
  }
}

template <typename T>
bool CrossCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  auto input1_data_addr = reinterpret_cast<T *>(inputs[0]->addr);
  int64_t tmp = 1;
  for (size_t i = 0; i < input1_shape_.size(); i++) {
    tmp = tmp * input1_shape_[i];
  }
  size_t input1_data_num = LongToSize(tmp);
  auto input2_data_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_data_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t total = input1_data_num / kNumber3;
  const size_t n = input1_shape_.size();
  std::vector<size_t> a_stride(n);
  int64_t stride_tmp = 1;
  for (int64_t i = static_cast<int64_t>(n - 1); i >= 0; i--) {
    a_stride[LongToSize(i)] = LongToSize(stride_tmp);
    stride_tmp *= input1_shape_[LongToSize(i)];
  }
  size_t input1_data_stride = a_stride[LongToSize(dim_)];
  std::vector<size_t> b_stride(n);
  stride_tmp = 1;
  for (int64_t i = static_cast<int64_t>(n - 1); i >= 0; i--) {
    b_stride[LongToSize(i)] = LongToSize(stride_tmp);
    stride_tmp *= input2_shape_[LongToSize(i)];
  }
  size_t input2_data_stride = b_stride[LongToSize(dim_)];
  std::vector<size_t> r_stride(n);
  stride_tmp = 1;
  for (int64_t i = static_cast<int64_t>(n - 1); i >= 0; i--) {
    r_stride[LongToSize(i)] = LongToSize(stride_tmp);
    stride_tmp *= output_shape_[LongToSize(i)];
  }
  size_t output_data_stride = r_stride[LongToSize(dim_)];
  const size_t pos = 2;
  auto cross_shard = [this, &a_stride, &b_stride, &r_stride, &output_data_addr, &input1_data_addr, &input2_data_addr,
                      &output_data_stride, &input1_data_stride, &input2_data_stride, &pos](size_t start, size_t end) {
    const size_t input1_data_dim = input1_shape_.size();
    std::vector<int64_t> position_in_dims(input1_data_dim);
    int64_t index_in_curr_dim = SizeToLong(start);
    int64_t input1_data_start = 0;
    int64_t input2_data_start = 0;
    int64_t output_data_start = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(input1_data_dim); i++) {
      if (i == static_cast<int64_t>(dim_)) {
        continue;
      }
      position_in_dims[LongToSize(i)] = index_in_curr_dim % input1_shape_[LongToSize(i)];
      input1_data_start += (index_in_curr_dim % input1_shape_[LongToSize(i)]) * SizeToLong(a_stride[LongToSize(i)]);
      input2_data_start += (index_in_curr_dim % input2_shape_[LongToSize(i)]) * SizeToLong(b_stride[LongToSize(i)]);
      output_data_start += (index_in_curr_dim % output_shape_[LongToSize(i)]) * SizeToLong(r_stride[LongToSize(i)]);
      index_in_curr_dim = index_in_curr_dim / input1_shape_[LongToSize(i)];
    }
    while (start < end) {
      output_data_addr[output_data_start + SizeToLong(kNumber0 * output_data_stride)] =
        static_cast<T>(input1_data_addr[input1_data_start + SizeToLong(kNumber1 * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(pos * input2_data_stride)] -
                       input1_data_addr[input1_data_start + SizeToLong(pos * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(kNumber1 * input2_data_stride)]);
      output_data_addr[output_data_start + SizeToLong(kNumber1 * output_data_stride)] =
        static_cast<T>(input1_data_addr[input1_data_start + SizeToLong(pos * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(kNumber0 * input2_data_stride)] -
                       input1_data_addr[input1_data_start + SizeToLong(kNumber0 * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(pos * input2_data_stride)]);
      output_data_addr[output_data_start + SizeToLong(pos * output_data_stride)] =
        static_cast<T>(input1_data_addr[input1_data_start + SizeToLong(kNumber0 * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(kNumber1 * input2_data_stride)] -
                       input1_data_addr[input1_data_start + SizeToLong(kNumber1 * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(kNumber0 * input2_data_stride)]);
      start++;
      for (size_t i = 0; i < input1_data_dim; i++) {
        if (i == static_cast<size_t>(dim_)) {
          continue;
        }
        position_in_dims[i]++;
        input1_data_start += SizeToLong(a_stride[i]);
        input2_data_start += SizeToLong(b_stride[i]);
        output_data_start += SizeToLong(r_stride[i]);
        if (position_in_dims[i] == input1_shape_[i] && i != (input1_shape_.size() - 1)) {
          input1_data_start -= position_in_dims[i] * SizeToLong(a_stride[i]);
          input2_data_start -= position_in_dims[i] * SizeToLong(b_stride[i]);
          output_data_start -= position_in_dims[i] * SizeToLong(r_stride[i]);
          position_in_dims[i] = 0;
        } else {
          break;
        }
      }
    }
  };
  if (total * sizeof(T) < kDataSizeThreshold) {
    cross_shard(0, total);
  } else {
    CPUKernelUtils::ParallelFor(cross_shard, total);
  }
  return true;
}

std::vector<KernelAttr> CrossCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr()
      .AddInputAttr(kNumberTypeComplex64)
      .AddInputAttr(kNumberTypeComplex64)
      .AddOutputAttr(kNumberTypeComplex64),
    KernelAttr()
      .AddInputAttr(kNumberTypeComplex128)
      .AddInputAttr(kNumberTypeComplex128)
      .AddOutputAttr(kNumberTypeComplex128)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Cross, CrossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
