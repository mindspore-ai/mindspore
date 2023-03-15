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

#include "plugin/device/gpu/kernel/arrays/im2col_gpu_kernel.h"
#include <algorithm>
#include <complex>
#include "mindspore/core/ops/im2col.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIm2ColInputsNum = 1;
constexpr size_t kIm2ColOutputsNum = 1;
constexpr int64_t kInt64Number2 = 2;
}  // namespace
template <typename T>
using Complex = mindspore::utils::Complex<T>;
using Complex64 = Complex<float>;
using Complex128 = Complex<double>;

bool Im2ColGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIm2ColInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIm2ColOutputsNum, kernel_name_);

  auto kernel_ptr = std::dynamic_pointer_cast<ops::Im2Col>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast HShrink ops failed!";
    return false;
  }
  ksizes_ = kernel_ptr->get_ksizes();
  strides_ = kernel_ptr->get_strides();
  dilations_ = kernel_ptr->get_dilations();
  pads_ = kernel_ptr->get_pads();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int Im2ColGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  x_shape_ = inputs[0]->GetShapeVector();
  y_shape_ = outputs[0]->GetShapeVector();
  return KRET_OK;
}

template <typename T>
bool Im2ColGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIm2ColInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIm2ColOutputsNum, kernel_name_);

  auto x = GetDeviceAddress<T>(inputs, kIndex0);
  auto y = GetDeviceAddress<T>(outputs, kIndex0);

  const int64_t batches = x_shape_[kIndex0];
  const int64_t x_channel = x_shape_[kIndex1];
  const int64_t x_height = x_shape_[kIndex2];
  const int64_t x_width = x_shape_[kIndex3];

  const int64_t y_out_plane = y_shape_[kIndex1] * y_shape_[kIndex2];
  const int64_t total_block = y_shape_[kIndex3];

  const int64_t kernel_height = ksizes_.front();
  MS_EXCEPTION_IF_ZERO("kernel_height", kernel_height);
  const int64_t kernel_width = ksizes_.back();
  MS_EXCEPTION_IF_ZERO("kernel_width", kernel_width);
  const int64_t stride_height = strides_.front();
  MS_EXCEPTION_IF_ZERO("stride_height", stride_height);
  const int64_t stride_width = strides_.back();
  MS_EXCEPTION_IF_ZERO("stride_width", stride_width);
  const int64_t dilation_height = dilations_.front();
  MS_EXCEPTION_IF_ZERO("dilation_height", dilation_height);
  const int64_t dilation_width = dilations_.back();
  MS_EXCEPTION_IF_ZERO("dilation_width", dilation_width);

  int64_t y_height{0}, y_width{0};
  int64_t pad_height = 0, pad_width = 0;
  if (!pads_.empty() && (pads_.size() <= kDim2 || pads_.size() == kDim4)) {
    pad_height = pads_.front();
    pad_width = pads_.back();
  } else {
    MS_EXCEPTION(ValueError) << "For 'Im2Col', the size of pads_ must be 1 or 2, but get " << pads_.size()
                             << "elements in pads_.";
  }
  y_height = (x_height + pad_height + pad_height - (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
  y_width = (x_width + pad_width + pad_width - (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;

  if (total_block != y_height * y_width) {
    MS_EXCEPTION(ValueError) << "For Im2Col, the output shape's last dim must be equal to y_height * y_width"
                             << "but got total_block = " << total_block << ", [y_height, y_width] = [" << y_height
                             << ", " << y_width << "].";
  }

  CudaIm2Col(batches, x_channel, x_height, x_width, y_out_plane, y_height, y_width, kernel_height, kernel_width,
             stride_height, stride_width, dilation_height, dilation_width, pad_height, pad_width, x, y, device_id_,
             reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaGetLastError(), "Im2Col kernel failed.");
  return true;
}

std::vector<std::pair<KernelAttr, Im2ColGpuKernelMod::Im2ColFunc>> Im2ColGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &Im2ColGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &Im2ColGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &Im2ColGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &Im2ColGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &Im2ColGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &Im2ColGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &Im2ColGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &Im2ColGpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &Im2ColGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &Im2ColGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &Im2ColGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &Im2ColGpuKernelMod::LaunchKernel<Complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &Im2ColGpuKernelMod::LaunchKernel<Complex128>},
};

std::vector<KernelAttr> Im2ColGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, Im2ColGpuKernelMod::Im2ColFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Im2Col, Im2ColGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
