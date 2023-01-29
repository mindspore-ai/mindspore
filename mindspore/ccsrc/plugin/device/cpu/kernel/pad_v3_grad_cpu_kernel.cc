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

#include "plugin/device/cpu/kernel/pad_v3_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/op_name.h"
#include "mindspore/core/ops/grad/pad_v3_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kPadV3Grad = "PadV3Grad";
constexpr const size_t kInputsNum = 2;
constexpr const size_t kOutputsNum = 1;
constexpr int64_t k3DNum = 6;
constexpr int64_t k2DNum = 4;
constexpr int64_t k1DNum = 2;
constexpr int64_t kpad_l = 0;
constexpr int64_t kpad_t = 2;
constexpr int64_t kpad_f = 4;
constexpr int64_t kpad_r = 1;
constexpr int64_t kpad_d = 3;
constexpr int64_t kpad_b = 5;
constexpr int64_t kwidth = 1;
constexpr int64_t kheight = 2;
constexpr int64_t kchannel = 3;
constexpr int64_t kInput1Dim = 3;
constexpr int64_t kInput2Dim = 4;
constexpr int64_t kInput3Dim = 5;
constexpr int64_t k2Num = 2;
constexpr int64_t padding_pos_2 = 2;
constexpr int64_t padding_pos_3 = 3;
constexpr int64_t padding_pos_4 = 4;
const std::vector<std::string> mode_list = {ops::kReflect, ops::kEdge, ops::kCircular};
}  // namespace

bool PadV3GradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto kernel_ptr = std::dynamic_pointer_cast<ops::PadV3Grad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  mode_ = kernel_ptr->get_mode();
  const bool is_mode_available = std::find(mode_list.begin(), mode_list.end(), mode_) != mode_list.end();
  if (is_mode_available == false) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'mode' should be 'reflect', 'edge' or 'circular', but got "
                  << mode_;
    return false;
  }

  paddings_contiguous_ = kernel_ptr->get_paddings_contiguous();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  dtype_ = inputs[kIndex0]->GetDtype();
  return true;
}

int PadV3GradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  input_dim_ = SizeToLong(input_shape.size());
  input_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kIndex0]->GetDeviceShapeAdaptively();

  auto padding_shape = inputs[kIndex1]->GetShapeVector();
  // get padding_num
  if (padding_shape.size() != 1) {
    paddings_num_ = 1;
  } else {
    paddings_num_ = SizeToLong(padding_shape[0]);
  }
  return KRET_OK;
}

template <typename S>
bool PadV3GradCpuKernelMod::GetPaddings(const std::vector<AddressPtr> &inputs) {
  // get paddings
  auto paddings_arg = static_cast<S *>(inputs[1]->addr);
  if (paddings_num_ == 1) {
    paddings_num_ = k2Num * (input_dim_ - k2Num);
    for (int64_t i = 0; i < paddings_num_; ++i) {
      paddings_[i] = int64_t(*paddings_arg);
    }
  } else {
    for (int64_t i = 0; i < paddings_num_; ++i) {
      paddings_[i] = int64_t(*(paddings_arg + i));
    }
  }

  // get parallelSliceNum_
  for (int64_t i = 0; i < input_dim_ - paddings_num_ / k2Num; i++) {
    parallelSliceNum_ *= input_shape_[i];
  }

  if (paddings_contiguous_ == false && paddings_num_ == k3DNum) {
    std::vector<int64_t> tmp = paddings_;
    paddings_[1] = tmp[padding_pos_3];
    paddings_[padding_pos_2] = tmp[1];
    paddings_[padding_pos_3] = tmp[padding_pos_4];
    paddings_[padding_pos_4] = tmp[padding_pos_2];
  }

  if (paddings_contiguous_ == false && paddings_num_ == k2DNum) {
    std::vector<int64_t> tmp = paddings_;
    paddings_[1] = tmp[padding_pos_2];
    paddings_[padding_pos_2] = tmp[1];
  }
  return true;
}

template <typename T>
void PadV3GradCpuKernelMod::PadV3GradCompute(T *input, T *output, int64_t p) const {
  if (paddings_num_ == k1DNum) {
    PadV3GradCompute1D<T>(input, output, p);
  } else if (paddings_num_ == k2DNum) {
    for (int i = 0; i < input_h_; i++) {
      PadV3GradCompute2D<T>(input, output, p, i);
    }
  } else if (paddings_num_ == k3DNum) {
    for (int z = 0; z < input_c_; z++) {
      PadV3GradCompute3D<T>(input, output, p, z);
    }
  }
}

template <typename T>
void PadV3GradCpuKernelMod::PadV3GradCompute1D(T *input, T *output, int64_t p) const {
  for (int j = 0; j < input_w_; j++) {
    auto ip_x = IndexCalculate(pad_l_, pad_r_, j, output_w_, o_start_x_, i_start_x_);
    T *src_p = input + p * input_w_ + j;
    T *dest_p = output + p * output_w_ + ip_x;
    *dest_p += *src_p;
  }
}

template <typename T>
void PadV3GradCpuKernelMod::PadV3GradCompute2D(T *input, T *output, int64_t p, int64_t i) const {
  for (int j = 0; j < input_w_; j++) {
    auto ip_x = IndexCalculate(pad_l_, pad_r_, j, output_w_, o_start_x_, i_start_x_);
    auto ip_y = IndexCalculate(pad_t_, pad_d_, i, output_h_, o_start_y_, i_start_y_);
    T *src_p = input + p * input_w_ * input_h_ + i * input_w_ + j;
    T *dest_p = output + p * output_w_ * output_h_ + ip_y * output_w_ + ip_x;
    *dest_p += *src_p;
  }
}

template <typename T>
void PadV3GradCpuKernelMod::PadV3GradCompute3D(T *input, T *output, int64_t p, int64_t z) const {
  for (int i = 0; i < input_h_; i++) {
    for (int j = 0; j < input_w_; j++) {
      auto ip_x = IndexCalculate(pad_l_, pad_r_, j, output_w_, o_start_x_, i_start_x_);
      auto ip_y = IndexCalculate(pad_t_, pad_d_, i, output_h_, o_start_y_, i_start_y_);
      auto ip_z = IndexCalculate(pad_f_, pad_b_, z, output_c_, o_start_z_, i_start_z_);
      T *src_p = input + p * input_w_ * input_h_ * input_c_ + z * input_w_ * input_h_ + i * input_w_ + j;
      T *dest_p =
        output + p * output_w_ * output_h_ * output_c_ + ip_z * output_w_ * output_h_ + ip_y * output_w_ + ip_x;
      *dest_p += *src_p;
    }
  }
}

int64_t PadV3GradCpuKernelMod::IndexCalculate(int64_t pad_value, int64_t pad_end, int64_t now, int64_t output_value,
                                              int64_t o_start, int64_t i_start) const {
  int64_t ip = 0;
  if (now < pad_value) {
    if (mode_ == ops::kReflect) {
      ip = pad_value + pad_value - now;
    } else if (mode_ == ops::kEdge) {
      ip = pad_value;
    } else if (mode_ == ops::kCircular) {
      ip = output_value + now + std::min(int64_t(0), pad_end);
    }
  } else if (now >= pad_value && now < output_value + pad_value) {
    ip = now;
  } else {
    if (mode_ == ops::kReflect) {
      ip = (output_value + pad_value - 1) + (output_value + pad_value - 1) - now;
    } else if (mode_ == ops::kEdge) {
      ip = output_value + pad_value - 1;
    } else if (mode_ == ops::kCircular) {
      ip = now - output_value - std::min(int64_t(0), pad_value);
    }
  }
  ip = ip - o_start + i_start;
  return ip;
}

template <typename T, typename S>
bool PadV3GradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                         const std::vector<AddressPtr> &outputs) {
  if (!GetPaddings<S>(inputs)) {
    MS_LOG(EXCEPTION) << "get paddings failed";
  }
  output_w_ = output_shape_.end()[-kwidth];
  output_h_ = output_shape_.end()[-kheight];
  output_c_ = output_shape_.end()[-kchannel];
  input_w_ = input_shape_.end()[-kwidth];
  input_h_ = input_shape_.end()[-kheight];
  input_c_ = input_shape_.end()[-kchannel];

  i_start_x_ = std::max(int64_t(0), -paddings_[kpad_l]);
  i_start_y_ = std::max(int64_t(0), -paddings_[kpad_t]);
  i_start_z_ = std::max(int64_t(0), -paddings_[kpad_f]);
  o_start_x_ = std::max(int64_t(0), paddings_[kpad_l]);
  o_start_y_ = std::max(int64_t(0), paddings_[kpad_t]);
  o_start_z_ = std::max(int64_t(0), paddings_[kpad_f]);

  pad_l_ = paddings_[kpad_l];
  pad_t_ = paddings_[kpad_t];
  pad_f_ = paddings_[kpad_f];
  pad_r_ = paddings_[kpad_r];
  pad_d_ = paddings_[kpad_d];
  pad_b_ = paddings_[kpad_b];

  int64_t output_num_ = 1;
  for (int64_t i = 0; i < input_dim_; i++) {
    output_num_ *= output_shape_[i];
  }

  auto input = static_cast<T *>(inputs[0]->addr);
  auto output = static_cast<T *>(outputs[0]->addr);

  if (dtype_ == kNumberTypeComplex64 || dtype_ == kNumberTypeComplex128) {
    for (size_t i = 0; i < LongToSize(output_num_); ++i) {
      output[i] = static_cast<T>(0);
    }
  } else {
    if (memset_s(output, sizeof(T) * output_num_, 0, sizeof(T) * output_num_) != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', Failed to initialize output memory";
    }
  }

  auto task = [&](int64_t start, int64_t end) {
    for (int p = start; p < end; p++) {
      PadV3GradCompute<T>(input, output, p);
    }
  };
  ParallelLaunchAutoSearch(task, parallelSliceNum_, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, PadV3GradCpuKernelMod::SelectFunc>> PadV3GradCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
   &PadV3GradCpuKernelMod::LaunchKernel<float16, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &PadV3GradCpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &PadV3GradCpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
   &PadV3GradCpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
   &PadV3GradCpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &PadV3GradCpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &PadV3GradCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
   &PadV3GradCpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
   &PadV3GradCpuKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
   &PadV3GradCpuKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
   &PadV3GradCpuKernelMod::LaunchKernel<uint64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
   &PadV3GradCpuKernelMod::LaunchKernel<std::complex<float>, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
   &PadV3GradCpuKernelMod::LaunchKernel<std::complex<double>, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &PadV3GradCpuKernelMod::LaunchKernel<float16, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &PadV3GradCpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &PadV3GradCpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
   &PadV3GradCpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
   &PadV3GradCpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &PadV3GradCpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &PadV3GradCpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
   &PadV3GradCpuKernelMod::LaunchKernel<uint8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
   &PadV3GradCpuKernelMod::LaunchKernel<uint16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   &PadV3GradCpuKernelMod::LaunchKernel<uint32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
   &PadV3GradCpuKernelMod::LaunchKernel<uint64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
   &PadV3GradCpuKernelMod::LaunchKernel<std::complex<float>, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
   &PadV3GradCpuKernelMod::LaunchKernel<std::complex<double>, int32_t>}};

std::vector<KernelAttr> PadV3GradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SelectFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PadV3Grad, PadV3GradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
