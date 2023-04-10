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

#include "plugin/device/cpu/kernel/dilation2d_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kFilterIndex = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 1;
constexpr size_t kDimSize3 = 3;
constexpr size_t kDimSize4 = 4;
constexpr size_t kFormatNCHWIndexN = 0;
constexpr size_t kFormatNCHWIndexC = 1;
constexpr size_t kFormatNCHWIndexH = 2;
constexpr size_t kFormatNCHWIndexW = 3;
constexpr size_t kFormatCHWIndexH = 1;
constexpr size_t kFormatCHWIndexW = 2;
constexpr int64_t kValue2 = 2;
}  // namespace

bool Dilation2DCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Dilation2D>(base_operator);
  kernel_name_ = kernel_ptr->name();
  stride_ = kernel_ptr->get_stride();
  dilation_ = kernel_ptr->get_dilation();
  pad_mode_ = kernel_ptr->get_pad_mode();
  format_ = kernel_ptr->get_format();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

bool Dilation2DCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  return kernel_func_(this, inputs, workspace, outputs);
}

int Dilation2DCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  input_shape_ = inputs[kInputIndex]->GetShapeVector();
  filter_shape_ = inputs[kFilterIndex]->GetShapeVector();
  output_shape_ = outputs[kOutputIndex]->GetShapeVector();
  (void)CheckKernelParam();
  return KRET_OK;
}

template <typename T>
bool Dilation2DCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  T *input = static_cast<T *>(inputs[kInputIndex]->addr);
  T *filter = static_cast<T *>(inputs[kFilterIndex]->addr);
  T *output = static_cast<T *>(outputs[kOutputIndex]->addr);

  int64_t num_batch = input_shape_[kFormatNCHWIndexN];
  int64_t input_height = input_shape_[kFormatNCHWIndexH];
  int64_t input_width = input_shape_[kFormatNCHWIndexW];
  int64_t channel = input_shape_[kFormatNCHWIndexC];
  int64_t filter_height = filter_shape_[kFormatCHWIndexH];
  int64_t filter_width = filter_shape_[kFormatCHWIndexW];
  int64_t output_height = output_shape_[kFormatNCHWIndexH];
  int64_t output_width = output_shape_[kFormatNCHWIndexW];
  int64_t stride_height = stride_[kFormatNCHWIndexH];
  int64_t stride_width = stride_[kFormatNCHWIndexW];
  int64_t rate_height = dilation_[kFormatNCHWIndexH];
  int64_t rate_width = dilation_[kFormatNCHWIndexW];
  // default value for VALID pad mode
  int64_t pad_top = 0;
  int64_t pad_left = 0;

  if (pad_mode_.compare("SAME") == 0 || pad_mode_.compare("same") == 0) {
    int64_t pad_height = (output_height - 1) * stride_height + rate_height * (filter_height - 1) + 1 - input_height;
    pad_height = pad_height >= 0 ? pad_height : 0;
    int64_t pad_width = (output_width - 1) * stride_width + rate_width * (filter_width - 1) + 1 - input_width;
    pad_width = pad_width >= 0 ? pad_width : 0;
    pad_top = pad_height / kValue2;
    pad_left = pad_width / kValue2;
  }

  int64_t output_plane_size = output_height * output_width;

  for (int64_t n = 0; n < num_batch; ++n) {
    for (int64_t c = 0; c < channel; ++c) {
      for (int64_t pos = 0; pos < output_plane_size; ++pos) {
        int64_t pos_h = pos / output_width % output_height;
        int64_t pos_w = pos % output_width;
        int64_t h_start = pos_h * stride_height - pad_top;
        int64_t w_start = pos_w * stride_width - pad_left;

        T max_val = std::numeric_limits<T>::lowest();

        for (int64_t h = 0; h < filter_height; ++h) {
          int64_t h_in = h_start + h * rate_height;
          if (h_in >= 0 && h_in < input_height) {
            for (int64_t w = 0; w < filter_width; ++w) {
              int64_t w_in = w_start + w * rate_width;
              if (w_in >= 0 && w_in < input_width) {
                T val = input[LongToSize(w_in + input_width * (h_in + input_height * (c + channel * n)))] +
                        filter[LongToSize(w + filter_width * (h + filter_height * c))];
                if (val > max_val) {
                  max_val = val;
                }
              }
            }
          }
        }
        output[LongToSize((n * channel + c) * output_plane_size + pos)] = max_val;
      }
    }
  }
  return true;
}

const std::vector<std::pair<KernelAttr, Dilation2DCpuKernelMod::KernelRunFunc>> &Dilation2DCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, Dilation2DCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &Dilation2DCpuKernelMod::LaunchKernel<float16>},

    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &Dilation2DCpuKernelMod::LaunchKernel<float>},

    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &Dilation2DCpuKernelMod::LaunchKernel<double>},

    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &Dilation2DCpuKernelMod::LaunchKernel<int8_t>},

    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &Dilation2DCpuKernelMod::LaunchKernel<int16_t>},

    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &Dilation2DCpuKernelMod::LaunchKernel<int32_t>},

    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &Dilation2DCpuKernelMod::LaunchKernel<int64_t>},

    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &Dilation2DCpuKernelMod::LaunchKernel<uint8_t>},

    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &Dilation2DCpuKernelMod::LaunchKernel<uint16_t>},

    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &Dilation2DCpuKernelMod::LaunchKernel<uint32_t>},

    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &Dilation2DCpuKernelMod::LaunchKernel<uint64_t>}};

  return func_list;
}

bool Dilation2DCpuKernelMod::CheckKernelParam() {
  size_t input_shape_dims = input_shape_.size();
  size_t filter_shape_dims = filter_shape_.size();
  size_t output_shape_dims = output_shape_.size();
  size_t stride_dims = stride_.size();
  size_t dilation_dims = dilation_.size();
  if (input_shape_dims != kDimSize4) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'input_shape' must be equal to 4, but got "
                  << input_shape_dims << ".";
    return false;
  }
  if (filter_shape_dims != kDimSize3) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'filter_shape' must be equal to 3, but got "
                  << filter_shape_dims << ".";
    return false;
  }
  if (output_shape_dims != kDimSize4) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'output_shape' must be equal to 4, but got "
                  << output_shape_dims << ".";
    return false;
  }
  if (stride_dims != kDimSize4) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'stride' must be equal to 4, but got "
                  << stride_dims << ".";
    return false;
  }
  if (dilation_dims != kDimSize4) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'dilation' must be equal to 4, but got "
                  << dilation_dims << ".";
    return false;
  }
  if (pad_mode_ != "VALID" && pad_mode_ != "valid" && pad_mode_ != "SAME" && pad_mode_ != "same") {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', pad_mode_ must be VALID, valid, SAME or same, but got " << pad_mode_
                  << ".";
    return false;
  }
  if (format_ != "NCHW") {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', data_format must be NCHW, but got " << format_ << ".";
    return false;
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Dilation2D, Dilation2DCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
