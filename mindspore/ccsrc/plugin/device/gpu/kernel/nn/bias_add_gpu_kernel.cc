/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/bias_add_gpu_kernel.h"
#include <mindspore/core/abstract/utils.h>
#include <map>

namespace mindspore {
namespace kernel {
bool BiasAddGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  InitResource();
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[kIndex1]->GetDtype()));
  return true;
}

int BiasAddGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto x_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  auto num_dims = x_shape.size();
  is_null_input_ = CHECK_SHAPE_NULL(x_shape, kernel_name_, "input_x");
  constexpr size_t min_num_dims = 2;
  if (num_dims < min_num_dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input_x cannot be less than 2, but got "
                      << num_dims;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BiasAdd>(base_operator);
  auto format = kernel_ptr->get_format();
  auto format_str = format_str_list[format + 1];
  string::size_type pos = format_str.find("C");
  if (pos == std::string::npos || pos >= num_dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'C' character must be in 'format', but got " << format_str;
  }

  // Expand to 4 dims for cudnnSetTensorNdDescriptorEx.
  constexpr size_t four_4D = 4;
  size_t cudnn_dims = std::max(num_dims, four_4D);
  std::unique_ptr<int[]> x_dims = std::make_unique<int[]>(cudnn_dims);
  std::unique_ptr<int[]> b_dims = std::make_unique<int[]>(cudnn_dims);
  for (size_t i = 0; i < cudnn_dims; i++) {
    x_dims[i] = (i < num_dims) ? LongToInt(x_shape[i]) : 1;
    b_dims[i] = (i == pos) ? LongToInt(x_shape[i]) : 1;
  }

  auto input_device_format = inputs[kIndex0]->GetFormat();
  auto cudnn_cal_format = (input_device_format == Format::NHWC) ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptorEx(x_desc_, cudnn_cal_format, cudnn_data_type_, SizeToInt(cudnn_dims), x_dims.get()),
    "cudnnSetTensorNdDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptorEx(b_desc_, cudnn_cal_format, cudnn_data_type_, SizeToInt(cudnn_dims), b_dims.get()),
    "cudnnSetTensorNdDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetOpTensorDescriptor(op_desc_, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN),
    "cudnnSetOpTensorDescriptor failed");
  return KRET_OK;
}

template <typename T>
bool BiasAddGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  VARIABLE_NOT_USED(workspace);
  VARIABLE_NOT_USED(stream_ptr);
  if (is_null_input_) {
    return true;
  }
  T *x_addr = GetDeviceAddress<T>(inputs, 0);
  T *b_addr = GetDeviceAddress<T>(inputs, 1);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  try {
    const float alpha = 1;
    const float beta = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnOpTensor(cudnn_handle_, op_desc_, &alpha, x_desc_, x_addr, &alpha, b_desc_,
                                                      b_addr, &beta, x_desc_, output_addr),
                                        "cudnnOpTensor failed");
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Encountered an exception: " << e.what() << " when invoke cudnnOpTensor";
  }
  return true;
}

std::vector<std::pair<KernelAttr, BiasAddGpuKernelMod::BiasAddLaunchFunc>> BiasAddGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &BiasAddGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &BiasAddGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &BiasAddGpuKernelMod::LaunchKernel<int8_t>}};

std::vector<KernelAttr> BiasAddGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, BiasAddGpuKernelMod::BiasAddLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BiasAdd, BiasAddGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
