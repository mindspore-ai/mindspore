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
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/reverse_sequence_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t input_num_ = 2;
constexpr size_t output_num_ = 1;
constexpr int kIndex0 = 0;
constexpr int kIndex1 = 1;
constexpr int kResizeFuncIndex = 2;
constexpr auto kSeqDim = "seq_dim";
constexpr auto kBatchDim = "batch_dim";
}  // namespace

void ReverseSequenceCpuKernelMod::ComputeStrides(const std::vector<int64_t> &shape, int *strides,
                                                 const int ndim) const {
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

int ReverseSequenceCpuKernelMod::CalcCountPreAxis(const std::vector<int64_t> &shape, int64_t axis) const {
  int count = 1;
  for (int i = 0; i < axis; ++i) {
    count *= shape.at(i);
  }
  return count;
}

int ReverseSequenceCpuKernelMod::CalcCountAfterAxis(const std::vector<int64_t> &shape, int64_t axis) const {
  int count = 1;
  for (int i = axis + 1; i < static_cast<int>(shape.size()); ++i) {
    count *= shape.at(i);
  }
  return count;
}

template <typename T>
void ReverseSequenceCpuKernelMod::ResizeKernel(const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  input0_shape_ = inputs[kIndex0]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  ndim_ = static_cast<int64_t>(input0_shape_.size());
  int64_t less_axis;
  int64_t greater_axis;
  if (batch_dim_ > seq_dim_) {
    less_axis = seq_dim_;
    greater_axis = batch_dim_;
  } else {
    less_axis = batch_dim_;
    greater_axis = seq_dim_;
  }
  outer_count_ = CalcCountPreAxis(input0_shape_, less_axis);
  outer_stride_ = LongToInt(input0_shape_[LongToSize(less_axis)]) * CalcCountAfterAxis(input0_shape_, less_axis);
  inner_count_ = 1;
  for (int64_t i = less_axis + 1; i < greater_axis; ++i) {
    inner_count_ *= input0_shape_[i];
  }
  inner_stride_ = LongToInt(input0_shape_[LongToSize(greater_axis)]) * CalcCountAfterAxis(input0_shape_, greater_axis);
  total_data_size_ = static_cast<int>(sizeof(T));
  for (int64_t i = 0; i < ndim_; ++i) {
    total_data_size_ *= input0_shape_[i];
  }
  copy_byte_size_ = static_cast<int>(sizeof(T)) * CalcCountAfterAxis(input0_shape_, greater_axis);
}

template <typename T, typename S>
void ReverseSequenceCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  auto input0 = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto input1 = reinterpret_cast<S *>(inputs[kIndex1]->addr);
  auto output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  auto ret = memcpy_s(output, IntToSize(total_data_size_), input0, IntToSize(total_data_size_));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed. Error no: " << ret;
  }
  ComputeStrides(input0_shape_, input_stride_, ndim_);
  ComputeStrides(output_shape_, output_stride_, ndim_);
  auto task = [this, input0, input1, output](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      const T *in = input0 + i * outer_stride_;
      T *out = output + i * outer_stride_;
      for (int batch = 0; batch < input0_shape_[batch_dim_]; batch++) {
        const T *in_batch = in + batch * input_stride_[batch_dim_];
        T *out_batch = out + batch * output_stride_[batch_dim_];
        auto seq_length = *(input1 + batch);
        if (seq_length > input0_shape_[seq_dim_]) {
          return;
        }
        for (int n = 0; n < seq_length; ++n) {
          const T *in_seq = in_batch + (seq_length - 1 - n) * input_stride_[seq_dim_];
          T *out_seq = out_batch + n * output_stride_[seq_dim_];
          for (int j = 0; j < inner_count_; ++j) {
            auto ret =
              memcpy_s(out_seq + j * inner_stride_, copy_byte_size_, in_seq + j * inner_stride_, copy_byte_size_);
            if (ret != EOK) {
              MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed. Error no: " << ret;
            }
          }
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, outer_count_, this, &parallel_search_info_, pool_);
}

bool ReverseSequenceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = base_operator->name();
  seq_dim_ = GetValue<int64_t>(prim->GetAttr(kSeqDim));
  batch_dim_ = GetValue<int64_t>(prim->GetAttr(kBatchDim));

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = std::get<kIndex1>(func_list_[index]);
  resize_func_ = std::get<kResizeFuncIndex>(func_list_[index]);
  return true;
}

int ReverseSequenceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  resize_func_(this, inputs, outputs);
  return KRET_OK;
}

bool ReverseSequenceCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num_, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num_, kernel_name_);

  kernel_func_(this, inputs, outputs);
  return true;
}

std::vector<std::tuple<KernelAttr, ReverseSequenceCpuKernelMod::KernelFunc, ReverseSequenceCpuKernelMod::ResizeFunc>>
  ReverseSequenceCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
     &ReverseSequenceCpuKernelMod::LaunchKernel<int8_t, int32_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
     &ReverseSequenceCpuKernelMod::LaunchKernel<int16_t, int32_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &ReverseSequenceCpuKernelMod::LaunchKernel<int32_t, int32_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &ReverseSequenceCpuKernelMod::LaunchKernel<int64_t, int32_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &ReverseSequenceCpuKernelMod::LaunchKernel<uint8_t, int32_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
     &ReverseSequenceCpuKernelMod::LaunchKernel<uint16_t, int32_t>,
     &ReverseSequenceCpuKernelMod::ResizeKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
     &ReverseSequenceCpuKernelMod::LaunchKernel<uint32_t, int32_t>,
     &ReverseSequenceCpuKernelMod::ResizeKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
     &ReverseSequenceCpuKernelMod::LaunchKernel<uint64_t, int32_t>,
     &ReverseSequenceCpuKernelMod::ResizeKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &ReverseSequenceCpuKernelMod::LaunchKernel<float, int32_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &ReverseSequenceCpuKernelMod::LaunchKernel<double, int32_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
     &ReverseSequenceCpuKernelMod::LaunchKernel<complex64, int32_t>,
     &ReverseSequenceCpuKernelMod::ResizeKernel<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex128),
     &ReverseSequenceCpuKernelMod::LaunchKernel<complex128, int32_t>,
     &ReverseSequenceCpuKernelMod::ResizeKernel<complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     &ReverseSequenceCpuKernelMod::LaunchKernel<bool, int32_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
     &ReverseSequenceCpuKernelMod::LaunchKernel<int8_t, int64_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
     &ReverseSequenceCpuKernelMod::LaunchKernel<int16_t, int64_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &ReverseSequenceCpuKernelMod::LaunchKernel<int32_t, int64_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &ReverseSequenceCpuKernelMod::LaunchKernel<int64_t, int64_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &ReverseSequenceCpuKernelMod::LaunchKernel<uint8_t, int64_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
     &ReverseSequenceCpuKernelMod::LaunchKernel<uint16_t, int64_t>,
     &ReverseSequenceCpuKernelMod::ResizeKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
     &ReverseSequenceCpuKernelMod::LaunchKernel<uint32_t, int64_t>,
     &ReverseSequenceCpuKernelMod::ResizeKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
     &ReverseSequenceCpuKernelMod::LaunchKernel<uint64_t, int64_t>,
     &ReverseSequenceCpuKernelMod::ResizeKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &ReverseSequenceCpuKernelMod::LaunchKernel<float, int64_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &ReverseSequenceCpuKernelMod::LaunchKernel<double, int64_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
     &ReverseSequenceCpuKernelMod::LaunchKernel<complex64, int64_t>,
     &ReverseSequenceCpuKernelMod::ResizeKernel<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &ReverseSequenceCpuKernelMod::LaunchKernel<complex128, int64_t>,
     &ReverseSequenceCpuKernelMod::ResizeKernel<complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     &ReverseSequenceCpuKernelMod::LaunchKernel<bool, int64_t>, &ReverseSequenceCpuKernelMod::ResizeKernel<bool>}};

std::vector<KernelAttr> ReverseSequenceCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::tuple<KernelAttr, KernelFunc, ResizeFunc> &tuple_item) { return std::get<kIndex0>(tuple_item); });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ReverseSequence, ReverseSequenceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
