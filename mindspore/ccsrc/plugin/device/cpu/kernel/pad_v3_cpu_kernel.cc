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

#include "plugin/device/cpu/kernel/pad_v3_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kPadV3 = "PadV3";
constexpr const size_t kConstantInputsNum = 3;
constexpr const size_t kOtherInputsNum = 2;
constexpr const size_t kOutputsNum = 1;
constexpr int64_t kInput3D = 3;
constexpr int64_t kInput4D = 4;
constexpr int64_t kInput5D = 5;
constexpr int64_t kPadding1D = 2;
constexpr int64_t kPadding2D = 4;
constexpr int64_t kPadding3D = 6;
constexpr int64_t kNum2 = 2;
constexpr int64_t kNum3 = 3;
constexpr int64_t kNum4 = 4;
const std::vector<std::string> mode_list = {"constant", "reflect", "edge"};
}  // namespace

void PadV3CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  mode_ = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "mode");
  const bool is_mode_available = std::find(mode_list.begin(), mode_list.end(), mode_) != mode_list.end();
  if (is_mode_available == false) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'mode' should be 'constant', 'reflect' or 'edge', but got "
                  << mode_;
  }
  paddings_contiguous_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "paddings_contiguous");
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  input_dim_ = SizeToLong(input_shape.size());
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto padding_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (padding_shape.size() != 1) {
    paddings_num_ = 1;
  } else {
    paddings_num_ = SizeToLong(padding_shape[0]);
  }
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename S>
bool PadV3CpuKernelMod::GetPaddings(const std::vector<AddressPtr> &inputs) {
  auto paddings_arg = static_cast<S *>(inputs[1]->addr);
  paddings_ = std::vector<int64_t>(input_dim_ * kNum2, 0);
  for (int64_t i = 0; i < paddings_num_; ++i) {
    paddings_[i] = int64_t(*(paddings_arg + i));
  }
  if (paddings_contiguous_ == false) {
    std::vector<int64_t> tmp = paddings_;
    for (int64_t i = 0; i < paddings_num_; ++i) {
      if (i % kNum2 == 0) {
        paddings_[i] = tmp[i / kNum2];
      } else {
        paddings_[i] = tmp[(i + paddings_num_) / kNum2];
      }
    }
  }
  if (paddings_num_ == 1) {
    for (int64_t i = 0; i < kNum2 * (input_dim_ - kNum2); ++i) {
      paddings_[i] = int64_t(*paddings_arg);
    }
    paddings_num_ = kNum2 * (input_dim_ - kNum2);
  }
  parallelSliceNum_ = 1;
  for (int64_t i = 0; i < input_dim_ - paddings_num_ / kNum2; ++i) {
    parallelSliceNum_ *= input_shape_[i];
  }
  for (int64_t i = 0; i < paddings_num_ / kNum2; ++i) {
    output_shape_.end()[-(i + 1)] += (paddings_[i * kNum2] + paddings_[i * kNum2 + 1]);
  }
  return true;
}

template <typename T>
void PadV3CpuKernelMod::ConstantModeCompute(T *input_ptr, T *output_ptr, T constant_values) {
  int64_t output_num = 1;
  for (int64_t i = 0; i < input_dim_; ++i) {
    output_num *= output_shape_[i];
  }
  int64_t input_num = 1;
  std::vector<int64_t> input_strides(input_dim_, 0);
  std::vector<int64_t> output_strides(input_dim_, 0);
  input_strides[input_dim_ - 1] = 1;
  output_strides[input_dim_ - 1] = 1;
  for (int64_t i = input_dim_ - 1; i >= 1; --i) {
    input_strides[i - 1] = input_strides[i] * input_shape_[i];
    output_strides[i - 1] = output_strides[i] * output_shape_[i];
  }
  std::vector<int64_t> offsets(input_dim_, 0);
  std::vector<int64_t> extents(input_dim_, 0);
  for (int64_t i = input_dim_ - 1; i >= 0; --i) {
    extents[i] = input_shape_[i];
    if (paddings_[i * kNum2] < 0) {
      extents[i] += paddings_[i * kNum2];
      offsets[i] = -paddings_[i * kNum2];
      paddings_[i * kNum2] = 0;
    }
    if (paddings_[i * kNum2 + 1] < 0) {
      extents[i] += paddings_[i * kNum2 + 1];
      paddings_[i * kNum2 + 1] = 0;
    }
    input_shape_[i] = extents[i];
    input_num *= input_shape_[i];
  }
  std::vector<T> input_values;
  for (int64_t i = 0; i < input_num; ++i) {
    int64_t k = i;
    int64_t p = 0;
    for (int64_t j = input_dim_ - 1; j >= 0; --j) {
      p += (offsets[j] + (k % extents[j])) * input_strides[j];
      k /= extents[j];
    }
    input_values.push_back(*(input_ptr + p));
  }
  for (int64_t i = 0; i < output_num; ++i) {
    *(output_ptr + i) = constant_values;
  }
  if (input_dim_ == 1) {
    for (int64_t i = 0; i < input_num; ++i) {
      *(output_ptr + paddings_[0] + i) = input_values[i];
    }
  } else {
    std::vector<int64_t> i_inx_add(input_dim_, 0);
    std::vector<int64_t> o_inx_add(input_dim_, 0);
    i_inx_add[input_dim_ - 1] = output_strides[input_dim_ - 1] * paddings_[kNum2 * (input_dim_ - 1)];
    o_inx_add[input_dim_ - 1] = output_strides[input_dim_ - 1] * paddings_[kNum2 * (input_dim_ - 1) + 1];
    for (int64_t i = input_dim_ - 1; i >= 1; --i) {
      i_inx_add[i - 1] = i_inx_add[i] + output_strides[i - 1] * paddings_[kNum2 * (i - 1)];
      o_inx_add[i - 1] = o_inx_add[i] + output_strides[i - 1] * paddings_[kNum2 * (i - 1) + 1];
    }
    int64_t i_inx = 0;
    int64_t o_inx = i_inx_add[0];
    std::vector<int64_t> pos(input_dim_ - 1, 0);
    while (i_inx < input_num) {
      for (int64_t i = 0; i < input_shape_[input_dim_ - 1]; ++i) {
        *(output_ptr + o_inx + i) = input_values[i_inx + i];
      }
      pos[input_dim_ - kNum2] += 1;
      int64_t dep = input_dim_ - 1;
      for (int64_t i = input_dim_ - 2; i >= 0; --i) {
        if (i > 0 && pos[i] >= input_shape_[i]) {
          pos[i] -= input_shape_[i];
          pos[i - 1] += 1;
          dep = i;
        } else {
          break;
        }
      }
      o_inx += i_inx_add[dep] + o_inx_add[dep] + input_shape_[input_dim_ - 1];
      i_inx += input_shape_[input_dim_ - 1];
    }
  }
}

template <typename T>
void PadV3CpuKernelMod::OtherModeCompute(T *input_ptr, T *output_ptr, int64_t p) const {
  if (paddings_num_ == kPadding1D) {
    OtherModeCompute1D<T>(input_ptr, output_ptr, p);
  } else if (paddings_num_ == kPadding2D) {
    OtherModeCompute2D<T>(input_ptr, output_ptr, p);
  } else if (paddings_num_ == kPadding3D) {
    OtherModeCompute3D<T>(input_ptr, output_ptr, p);
  }
}

template <typename T>
void PadV3CpuKernelMod::OtherModeCompute1D(T *input_ptr, T *output_ptr, int64_t p) const {
  int64_t nplane = 0;
  int64_t input_w = input_shape_[kNum2];
  int64_t output_w = output_shape_.end()[-1];
  int64_t pad_l = paddings_[0];
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t o_start_x = std::max(int64_t(0), pad_l);
  for (int64_t j = 0; j < output_w; ++j) {
    auto ip_x = IndexCalculate(pad_l, j, input_w, o_start_x, i_start_x);
    T *dest_p = output_ptr + p * output_w * (nplane + 1) + j;
    T *src_p = input_ptr + +p * input_w * (nplane + 1) + ip_x;
    *dest_p = *src_p;
  }
}

template <typename T>
void PadV3CpuKernelMod::OtherModeCompute2D(T *input_ptr, T *output_ptr, int64_t p) const {
  int64_t pad_l = paddings_[0];
  int64_t pad_t = paddings_[kNum2];
  int64_t nplane = 0;
  int64_t input_h = input_shape_[kNum2];
  int64_t input_w = input_shape_[kNum3];
  int64_t output_h = input_h + pad_t + paddings_[kNum3];
  int64_t output_w = input_w + pad_l + paddings_[1];
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t i_start_y = std::max(int64_t(0), -pad_t);
  int64_t o_start_x = std::max(int64_t(0), pad_l);
  int64_t o_start_y = std::max(int64_t(0), pad_t);
  for (int64_t i = 0; i < output_h; ++i) {
    for (int64_t j = 0; j < output_w; ++j) {
      auto ip_x = IndexCalculate(pad_l, j, input_w, o_start_x, i_start_x);
      auto ip_y = IndexCalculate(pad_t, i, input_h, o_start_y, i_start_y);
      T *dest_p = output_ptr + p * output_w * output_h * (nplane + 1) + i * output_w + j;
      T *src_p = input_ptr + p * input_w * input_h * (nplane + 1) + ip_y * input_w + ip_x;
      *dest_p = *src_p;
    }
  }
}

template <typename T>
void PadV3CpuKernelMod::OtherModeCompute3D(T *input_ptr, T *output_ptr, int64_t p) const {
  int64_t pad_l = paddings_[0];
  int64_t pad_t = paddings_[kNum2];
  int64_t pad_f = paddings_[kNum4];
  int64_t nplane = 0;
  int64_t input_d = input_shape_[kNum2];
  int64_t input_h = input_shape_[kNum3];
  int64_t input_w = input_shape_[kNum4];
  int64_t output_d = output_shape_[kNum2];
  int64_t output_h = output_shape_[kNum3];
  int64_t output_w = output_shape_[kNum4];
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t i_start_y = std::max(int64_t(0), -pad_t);
  int64_t i_start_z = std::max(int64_t(0), -pad_f);
  int64_t o_start_x = std::max(int64_t(0), pad_l);
  int64_t o_start_y = std::max(int64_t(0), pad_t);
  int64_t o_start_z = std::max(int64_t(0), pad_f);
  for (int64_t k = 0; k < output_d; ++k) {
    for (int64_t j = 0; j < output_h; ++j) {
      for (int64_t i = 0; i < output_w; ++i) {
        auto ip_x = IndexCalculate(pad_l, i, input_w, o_start_x, i_start_x);
        auto ip_y = IndexCalculate(pad_t, j, input_h, o_start_y, i_start_y);
        auto ip_z = IndexCalculate(pad_f, k, input_d, o_start_z, i_start_z);
        T *dest_p =
          output_ptr + p * output_w * output_h * output_d * (nplane + 1) + k * output_w * output_h + j * output_w + i;
        T *src_p =
          input_ptr + p * input_w * input_h * input_d * (nplane + 1) + ip_z * input_w * input_h + ip_y * input_w + ip_x;
        *dest_p = *src_p;
      }
    }
  }
}

int64_t PadV3CpuKernelMod::IndexCalculate(int64_t pad_value, int64_t now, int64_t input_value, int64_t o_start,
                                          int64_t i_start) const {
  int64_t ip = 0;
  if (now < pad_value) {
    if (mode_ == "reflect") {
      ip = pad_value + pad_value - now;
    } else if (mode_ == "edge") {
      ip = pad_value;
    }
  } else if (now >= pad_value && now < input_value + pad_value) {
    ip = now;
  } else {
    if (mode_ == "reflect") {
      ip = (input_value + pad_value - 1) + (input_value + pad_value - 1) - now;
    } else if (mode_ == "edge") {
      ip = input_value + pad_value - 1;
    }
  }
  ip = ip - o_start + i_start;
  return ip;
}

template <typename T, typename S>
bool PadV3CpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs) {
  if (mode_ == "constant" || inputs.size() == kConstantInputsNum) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kConstantInputsNum, kernel_name_);
  } else {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kOtherInputsNum, kernel_name_);
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  if (!GetPaddings<S>(inputs)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', get paddings failed";
  }
  auto input_ptr = static_cast<T *>(inputs[0]->addr);
  auto output_ptr = static_cast<T *>(outputs[0]->addr);
  if (mode_ == "constant") {
    T constant_values = *(static_cast<T *>(inputs[2]->addr));
    for (int64_t i = 0; i < input_dim_ / kNum2; ++i) {
      int64_t u = paddings_[i * kNum2];
      int64_t v = paddings_[i * kNum2 + 1];
      paddings_[i * kNum2] = paddings_[kNum2 * (input_dim_ - i - 1)];
      paddings_[i * kNum2 + 1] = paddings_[kNum2 * (input_dim_ - i - 1) + 1];
      paddings_[kNum2 * (input_dim_ - i - 1)] = u;
      paddings_[kNum2 * (input_dim_ - i - 1) + 1] = v;
    }
    ConstantModeCompute<T>(input_ptr, output_ptr, constant_values);
  } else {
    auto task = [&](int64_t start, int64_t end) {
      for (int64_t p = start; p < end; ++p) {
        OtherModeCompute<T>(input_ptr, output_ptr, p);
      }
    };
    ParallelLaunchAutoSearch(task, parallelSliceNum_, this, &parallel_search_info_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, PadV3CpuKernelMod::SelectFunc>> PadV3CpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
   &PadV3CpuKernelMod::LaunchKernel<float16, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &PadV3CpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &PadV3CpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
   &PadV3CpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
   &PadV3CpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &PadV3CpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &PadV3CpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
   &PadV3CpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
   &PadV3CpuKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
   &PadV3CpuKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
   &PadV3CpuKernelMod::LaunchKernel<uint64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
   &PadV3CpuKernelMod::LaunchKernel<std::complex<float>, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
   &PadV3CpuKernelMod::LaunchKernel<std::complex<double>, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &PadV3CpuKernelMod::LaunchKernel<float16, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &PadV3CpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &PadV3CpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
   &PadV3CpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
   &PadV3CpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &PadV3CpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &PadV3CpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
   &PadV3CpuKernelMod::LaunchKernel<uint8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
   &PadV3CpuKernelMod::LaunchKernel<uint16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   &PadV3CpuKernelMod::LaunchKernel<uint32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
   &PadV3CpuKernelMod::LaunchKernel<uint64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
   &PadV3CpuKernelMod::LaunchKernel<std::complex<float>, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
   &PadV3CpuKernelMod::LaunchKernel<std::complex<double>, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &PadV3CpuKernelMod::LaunchKernel<float16, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &PadV3CpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &PadV3CpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &PadV3CpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &PadV3CpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &PadV3CpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &PadV3CpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &PadV3CpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &PadV3CpuKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &PadV3CpuKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &PadV3CpuKernelMod::LaunchKernel<uint64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64),
   &PadV3CpuKernelMod::LaunchKernel<std::complex<float>, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128),
   &PadV3CpuKernelMod::LaunchKernel<std::complex<double>, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &PadV3CpuKernelMod::LaunchKernel<float16, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &PadV3CpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &PadV3CpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &PadV3CpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &PadV3CpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &PadV3CpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &PadV3CpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &PadV3CpuKernelMod::LaunchKernel<uint8_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &PadV3CpuKernelMod::LaunchKernel<uint16_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &PadV3CpuKernelMod::LaunchKernel<uint32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &PadV3CpuKernelMod::LaunchKernel<uint64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64),
   &PadV3CpuKernelMod::LaunchKernel<std::complex<float>, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128),
   &PadV3CpuKernelMod::LaunchKernel<std::complex<double>, int32_t>}};

std::vector<KernelAttr> PadV3CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SelectFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PadV3, PadV3CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
