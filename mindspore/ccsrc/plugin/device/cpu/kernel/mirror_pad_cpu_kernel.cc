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

#include "plugin/device/cpu/kernel/mirror_pad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "ops/mirror_pad.h"

namespace mindspore {
namespace kernel {
namespace {
// preset size of paddings
constexpr int64_t MAX_PADDINGS = 5;
constexpr int64_t PADDING_SIZE = 2;
constexpr int64_t MODE_REFLECT = 1;
constexpr int64_t MODE_SYMMETRIC = 0;
constexpr size_t kTwo = 2;
constexpr size_t kInputNum = 2;
}  // namespace

template <typename T>
bool process_dim_one(T *outputs_addr, T *inputs_addr, int64_t *paddings, const int64_t input_elements,
                     const int64_t mode) {
  int ret =
    memcpy_s(outputs_addr, paddings[0] * int64_t(sizeof(T)), inputs_addr + mode, paddings[0] * int64_t(sizeof(T)));
  if (ret != EOK) {
    MS_EXCEPTION(TypeError) << "memcpy_s error, errorno(" << ret << ")";
  }
  ret = memcpy_s(outputs_addr + paddings[0] + input_elements, paddings[1] * int64_t(sizeof(T)),
                 inputs_addr + input_elements - paddings[1] - mode, paddings[1] * int64_t(sizeof(T)));
  if (ret != EOK) {
    MS_EXCEPTION(TypeError) << "memcpy_s error, errorno(" << ret << ")";
  }
  ret = memcpy_s(outputs_addr + paddings[0], input_elements * int64_t(sizeof(T)), inputs_addr,
                 input_elements * int64_t(sizeof(T)));
  if (ret != EOK) {
    MS_EXCEPTION(TypeError) << "memcpy_s error, errorno(" << ret << ")";
  }
  std::reverse(outputs_addr, outputs_addr + paddings[0]);
  std::reverse(outputs_addr + paddings[0] + input_elements, outputs_addr + paddings[0] + input_elements + paddings[1]);
  return true;
}

template <typename T>
void extract_paddings(const T *paddings_arg, int64_t padd_dim, int64_t *extracted_paddings) {
  for (int64_t i = 0; i < padd_dim; i++) {
    extracted_paddings[i * PADDING_SIZE] = int64_t(paddings_arg[i * PADDING_SIZE]);
    extracted_paddings[i * PADDING_SIZE + 1] = int64_t(paddings_arg[i * PADDING_SIZE + 1]);
  }
}

void CheckPaddingValue(int64_t *paddings, const std::vector<int64_t> &input_shape, const int64_t mode) {
  int64_t input_shape_size = static_cast<int64_t>(input_shape.size());
  for (int64_t i = 0; i < input_shape_size * PADDING_SIZE; i++) {
    if (paddings[i] < 0) {
      MS_LOG(EXCEPTION) << "For 'MirrorPad', all elements of paddings must be >= 0.";
    }
    if (mode == MODE_SYMMETRIC) {
      if (paddings[i] > static_cast<int64_t>(input_shape[i / PADDING_SIZE])) {
        MS_LOG(EXCEPTION) << "For 'MirrorPad', paddings must be no greater than the dimension size: " << paddings[i]
                          << " greater than " << static_cast<int64_t>(input_shape[i / PADDING_SIZE]);
      }
    } else if (mode == MODE_REFLECT) {
      if (paddings[i] >= static_cast<int64_t>(input_shape[i / PADDING_SIZE])) {
        MS_LOG(EXCEPTION) << "For 'MirrorPad', paddings must be no greater than the dimension size: " << paddings[i]
                          << " not less than " << static_cast<int64_t>(input_shape[i / PADDING_SIZE]);
      }
    }
  }
}

bool MirrorPadCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MirrorPad>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "cast ExtractVolumePatches ops failed!";
  }
  kernel_name_ = kernel_ptr->name();
  size_t input_num = inputs.size();
  if (input_num != kInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num;
  }
  size_t output_num = outputs.size();
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }
  std::string mode = kernel_ptr->get_mode();
  if (mode == "REFLECT") {
    mode_ = 1;
  } else if (mode == "SYMMETRIC") {
    mode_ = 0;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'mode' must be 'REFLECT' or 'SYMMETRIC', but got " << mode;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int MirrorPadCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  ShapeVector input_shape = inputs[0]->GetShapeVector();
  dims_ = int64_t(input_shape.size());

  input_shape_.clear();
  input_elements_ = 1;
  for (int64_t i = 0; i < dims_; ++i) {
    input_shape_.push_back(input_shape[i]);
    input_elements_ *= input_shape_[i];
  }
  auto padding_shape = inputs[1]->GetShapeVector();
  num_paddings_ = padding_shape[0];

  auto output_shape = outputs[0]->GetShapeVector();
  output_shape_.clear();
  output_elements_ = 1;
  for (int64_t i = 0; i < dims_; ++i) {
    output_shape_.push_back(output_shape[i]);
    output_elements_ *= output_shape_[i];
  }
  return static_cast<int>(KRET_OK);
}

template <typename T1, typename T2>
bool MirrorPadCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &workspace,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto inputs_addr = reinterpret_cast<T1 *>(inputs[0]->addr);
  auto *paddings_arg = reinterpret_cast<T2 *>(inputs[1]->addr);
  auto outputs_addr = reinterpret_cast<T1 *>(outputs[0]->addr);

  const int64_t padd_dim = num_paddings_;
  const int64_t mode = mode_;
  const int64_t element_size = int64_t(sizeof(T1));

  int64_t paddings[MAX_PADDINGS * PADDING_SIZE];  // local and fixed size to keep in registers
  for (int64_t i = 0; i < MAX_PADDINGS * PADDING_SIZE; i++) {
    paddings[i] = 0;
  }
  extract_paddings(paddings_arg, padd_dim, paddings);
  // quickly solve one-dimensional simple cases
  if (dims_ == 1) {
    return process_dim_one<T1>(outputs_addr, inputs_addr, paddings, input_elements_, mode);
  }
  CheckPaddingValue(paddings, input_shape_, mode);
  // solve other situations
  std::vector<int64_t> output_strides_(dims_, 0);
  output_strides_[dims_ - 1] = 1;
  for (int64_t i = dims_ - 1; i > 0; --i) {
    output_strides_[i - 1] = output_strides_[i] * output_shape_[i];
  }
  std::vector<std::pair<int64_t, int64_t>> index;
  index.resize(dims_);
  index[dims_ - 1] = std::make_pair(output_strides_[dims_ - 1] * paddings[(dims_ - 1) * PADDING_SIZE],
                                    output_strides_[dims_ - 1] * paddings[(dims_ - 1) * PADDING_SIZE + 1]);
  for (int64_t i = dims_ - 1; i > 0; --i) {
    index[i - 1].first = index[i].first + output_strides_[i - 1] * paddings[(i - 1) * PADDING_SIZE];
    index[i - 1].second = index[i].second + output_strides_[i - 1] * paddings[(i - 1) * PADDING_SIZE + 1];
  }
  std::vector<int64_t> output_pos;
  output_pos.resize(input_elements_ / input_shape_[dims_ - 1]);
  std::vector<int64_t> tmp_pos;
  int64_t copy_size = element_size * input_shape_[dims_ - 1];
  const size_t input_length = IntToSize(input_elements_);
  for (size_t i = 0; i < input_length; i += input_shape_[dims_ - 1]) {
    std::vector<int64_t> pos(dims_, 0);
    auto idx = i / input_shape_[dims_ - 1];
    for (int j = dims_ - 2; j >= 0; --j) {
      if (idx == 0) {
        break;
      }
      pos[j] = idx % input_shape_[j];
      idx /= input_shape_[j];
    }
    int64_t output_index = 0;
    for (size_t j = 0; j < pos.size(); j++) {
      output_index += (pos[j] + paddings[j * PADDING_SIZE]) * output_strides_[j];
    }
    int ret = memcpy_s(outputs_addr + output_index, copy_size, inputs_addr + i, copy_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    }
    output_pos[i / input_shape_[dims_ - 1]] = output_index;
  }
  for (int64_t i = dims_ - 1; i >= 0; --i) {
    int64_t block_size = output_strides_[i];
    copy_size = block_size * element_size;
    const size_t length = IntToSize(output_pos.size());
    auto input_shape = input_shape_;
    for (size_t j = 0; j < length; j++) {
      auto item = output_pos[j];
      T1 *base_output_ptr1 = outputs_addr + item;
      for (int64_t cnt = 1; cnt <= paddings[i * PADDING_SIZE]; ++cnt) {
        int ret = memcpy_s(base_output_ptr1 - cnt * block_size, copy_size,
                           base_output_ptr1 + (cnt - 1 + mode) * block_size, copy_size);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
        }
      }
      T1 *base_output_ptr2 = outputs_addr + item + input_shape[i] * block_size;
      for (int64_t cnt = 1; cnt <= paddings[i * PADDING_SIZE + 1]; ++cnt) {
        int ret = memcpy_s(base_output_ptr2 + (cnt - 1) * block_size, copy_size,
                           base_output_ptr2 - (cnt + mode) * block_size, copy_size);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
        }
      }
      bool invalid_pos = i > 0 && j % input_shape[i - 1] == 0;
      if (invalid_pos) {
        tmp_pos.push_back(item - paddings[i * PADDING_SIZE] * block_size);
      }
    }
    output_pos.clear();
    output_pos.resize(tmp_pos.size());
    (void)std::copy(tmp_pos.begin(), tmp_pos.end(), output_pos.begin());
    tmp_pos.clear();
  }
  return true;
}

using KernelRunFunc = MirrorPadCpuKernelMod::KernelRunFunc;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &MirrorPadCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &MirrorPadCpuKernelMod::LaunchKernel<float16, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &MirrorPadCpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &MirrorPadCpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
     &MirrorPadCpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
     &MirrorPadCpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &MirrorPadCpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &MirrorPadCpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &MirrorPadCpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
     &MirrorPadCpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
     &MirrorPadCpuKernelMod::LaunchKernel<std::complex<float>, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &MirrorPadCpuKernelMod::LaunchKernel<std::complex<double>, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     &MirrorPadCpuKernelMod::LaunchKernel<bool, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &MirrorPadCpuKernelMod::LaunchKernel<float16, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &MirrorPadCpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &MirrorPadCpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
     &MirrorPadCpuKernelMod::LaunchKernel<int8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
     &MirrorPadCpuKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &MirrorPadCpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &MirrorPadCpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &MirrorPadCpuKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
     &MirrorPadCpuKernelMod::LaunchKernel<uint16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
     &MirrorPadCpuKernelMod::LaunchKernel<std::complex<float>, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex128),
     &MirrorPadCpuKernelMod::LaunchKernel<std::complex<double>, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     &MirrorPadCpuKernelMod::LaunchKernel<bool, int32_t>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MirrorPad, MirrorPadCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
