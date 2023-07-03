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

#include "plugin/device/gpu/kernel/nn/softmax_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "mindspore/core/ops/softmax.h"
#include "mindspore/core/ops/log_softmax.h"

namespace mindspore {
namespace kernel {
bool SoftmaxGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  constexpr size_t input_num = 1;
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
  auto input_data_type = inputs.at(kIndex0)->GetDtype();
  type_id_size_ = abstract::TypeIdSize(input_data_type);

  return true;
}

int SoftmaxGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  // input, workspace and output will be assign in InitSizeLists.
  ResetResource();
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(shape_),
                       [](const int64_t &value) { return LongToInt(value); });
  shape_size_ = input_shape.size();
  std::vector<int> axis;
  if (kernel_name_ == "LogSoftmax") {
    is_log_softmax_ = true;
    auto log_soft_max_ptr = std::dynamic_pointer_cast<ops::LogSoftmax>(base_operator);
    auto log_soft_axis = LongToInt(log_soft_max_ptr->get_axis());
    axis.push_back(log_soft_axis);
  } else {
    is_log_softmax_ = false;
    auto soft_max_ptr = std::dynamic_pointer_cast<ops::Softmax>(base_operator);
    auto axis_me = soft_max_ptr->get_axis();
    (void)std::transform(axis_me.begin(), axis_me.end(), std::back_inserter(axis),
                         [](const int64_t &value) { return LongToInt(value); });
  }

  // axis size must be 1
  if (axis.size() < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the length of 'axis' cannot be equal to 0, but got " << axis.size();
    return KRET_RESIZE_FAILED;
  }
  if (axis.size() > 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the length of 'axis' cannot be greater than 1, but got "
                  << axis.size();
    return KRET_RESIZE_FAILED;
  }

  // check axis value
  axis_acc_ = maybe_wrap_dim(axis[0], shape_size_);

  auto input_element_num =
    std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);

  if (input_element_num > 0) {
    // calculate outer and inner size
    for (size_t i = 0; i < axis_acc_; ++i) outer_size_ *= shape_[i];
    for (size_t i = axis_acc_ + 1; i < shape_.size(); ++i) inner_size_ *= shape_[i];
  }

  // calculate size
  input_size_ = type_id_size_ * input_element_num;
  output_size_ = input_size_;
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool SoftmaxGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(input_addr, false);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(output_addr, false);

  // kernel function
  if (is_log_softmax_) {
    Softmax<T, true>(input_addr, output_addr, shape_[axis_acc_], outer_size_, inner_size_, device_id_,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    Softmax<T, false>(input_addr, output_addr, shape_[axis_acc_], outer_size_, inner_size_, device_id_,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
  }

  return true;
}

std::vector<std::pair<KernelAttr, SoftmaxGpuKernelMod::SoftmaxGpuLaunchFunc>> SoftmaxGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &SoftmaxGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &SoftmaxGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &SoftmaxGpuKernelMod::LaunchKernel<half>},
};

std::vector<KernelAttr> SoftmaxGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SoftmaxGpuKernelMod::SoftmaxGpuLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Softmax, SoftmaxGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogSoftmax, SoftmaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
