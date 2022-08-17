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

#include "plugin/device/cpu/kernel/im2col_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <string>
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIm2ColInputsNum = 1;
constexpr size_t kIm2ColOutputsNum = 1;
constexpr int64_t kInt64Number2 = 2;
}  // namespace

void Im2ColCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);

  x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex0);
  y_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, kIndex0);
  y_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, kIndex0);

  ksizes_ = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, kAttrKsizes);
  strides_ = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, kAttrStrides);
  dilations_ = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, kAttrDilations);
  padding_mode_ = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttPaddingMode);
  pads_ = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, kAttrPads);
}

bool Im2ColCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIm2ColInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIm2ColOutputsNum, kernel_name_);
  bool res = false;
  switch (y_type_) {
    case kNumberTypeInt8: {
      res = LaunchKernel<int8_t>(inputs, outputs);
      break;
    }
    case kNumberTypeInt16: {
      res = LaunchKernel<int16_t>(inputs, outputs);
      break;
    }
    case kNumberTypeInt32: {
      res = LaunchKernel<int32_t>(inputs, outputs);
      break;
    }
    case kNumberTypeInt64: {
      res = LaunchKernel<int64_t>(inputs, outputs);
      break;
    }
    case kNumberTypeUInt8: {
      res = LaunchKernel<uint8_t>(inputs, outputs);
      break;
    }
    case kNumberTypeUInt16: {
      res = LaunchKernel<uint16_t>(inputs, outputs);
      break;
    }
    case kNumberTypeUInt32: {
      res = LaunchKernel<uint32_t>(inputs, outputs);
      break;
    }
    case kNumberTypeUInt64: {
      res = LaunchKernel<uint64_t>(inputs, outputs);
      break;
    }
    case kNumberTypeFloat16: {
      res = LaunchKernel<float16>(inputs, outputs);
      break;
    }
    case kNumberTypeFloat32: {
      res = LaunchKernel<float>(inputs, outputs);
      break;
    }
    case kNumberTypeFloat64: {
      res = LaunchKernel<double>(inputs, outputs);
      break;
    }
    default:
      MS_LOG(EXCEPTION)
        << "For '" << kernel_name_
        << "', the dtype of 'x' should be one of [float16, float32, float64, uin8, int8, uint16, int16, "
           "uint32, int32, uint64, int64], but got "
        << TypeIdLabel(y_type_) << ".";
      break;
  }
  return res;
}

template <typename T>
bool Im2ColCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  // init output data
  auto y_data_ptr = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  std::fill_n(y_data_ptr, CPUKernelUtils::CalcElementNum(y_shape_), T(0));

  int64_t batch_size = x_shape_[kIndex0];
  int64_t x_height = x_shape_[kIndex2];
  int64_t x_width = x_shape_[kIndex3];

  int64_t y_channel = y_shape_[kIndex1];
  int64_t y_height = y_shape_[kIndex2];
  int64_t y_width = y_shape_[kIndex3];

  int64_t kernel_height = ksizes_.front();
  MS_EXCEPTION_IF_ZERO("kernel_height", kernel_height);
  int64_t kernel_width = ksizes_.back();
  MS_EXCEPTION_IF_ZERO("kernel_width", kernel_width);
  int64_t stride_height = strides_.front();
  MS_EXCEPTION_IF_ZERO("stride_height", stride_height);
  int64_t stride_width = strides_.back();
  MS_EXCEPTION_IF_ZERO("stride_width", stride_width);
  int64_t dilation_height = dilations_.front();
  MS_EXCEPTION_IF_ZERO("dilation_height", dilation_height);
  int64_t dilation_width = dilations_.back();
  MS_EXCEPTION_IF_ZERO("dilation_width", dilation_width);

  // pad distance
  int64_t pad_height = 0;
  int64_t pad_width = 0;
  if (padding_mode_ == "CALCULATED") {
    if (!pads_.empty() && pads_.size() <= kDim2) {
      pad_height = pads_.front();
      pad_width = pads_.back();
    } else if (!pads_.empty() && pads_.size() == kDim4) {
      pad_height = pads_[kIndex0];
      pad_width = pads_[kIndex2];
    }
  } else if (padding_mode_ == "SAME") {
    pad_height = (kernel_height - 1) / kInt64Number2;
    pad_width = (kernel_width - 1) / kInt64Number2;
  }  // else VALID no padding

  auto x_4d = EigenTensor(x_shape_, inputs[kIndex0]->addr).tensor<T, kDim4>();
  auto y_4d = EigenTensor(y_shape_, outputs[kIndex0]->addr).tensor<T, kDim4>();

  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (int64_t c_col = 0; c_col < y_channel; ++c_col) {
      int64_t w_offset = c_col % kernel_width;
      int64_t h_offset = (c_col / kernel_width) % kernel_height;
      int64_t c_im = c_col / kernel_height / kernel_width;
      for (int64_t h_col = 0; h_col < y_height; ++h_col) {
        int64_t h_im = h_col * stride_height - pad_height + h_offset * dilation_height;
        for (int64_t w_col = 0; w_col < y_width; ++w_col) {
          int64_t w_im = w_col * stride_width - pad_width + w_offset * dilation_width;
          y_4d(batch_idx, c_col, h_col, w_col) = (h_im >= 0 && w_im >= 0 && h_im < x_height && w_im < x_width)
                                                   ? x_4d(batch_idx, c_im, h_im, w_im)
                                                   : static_cast<T>(0);
        }
      }
    }
  }
  return true;
}

std::vector<KernelAttr> Im2ColCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Im2Col, Im2ColCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
