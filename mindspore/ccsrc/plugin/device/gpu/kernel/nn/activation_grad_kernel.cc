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

#include "plugin/device/gpu/kernel/nn/activation_grad_kernel.h"
#include <memory>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_grad_impl.cuh"
namespace mindspore {
namespace kernel {
namespace {
constexpr auto kReLU6Grad = "ReLU6Grad";
constexpr auto kTanhGrad = "TanhGrad";
constexpr auto kEluGrad = "EluGrad";
constexpr auto kSigmoidGrad = "SigmoidGrad";
}  // namespace

std::map<std::string, std::vector<std::pair<KernelAttr, ActivationGradGpuKernelMod::ActivationGradFunc>>>
  ActivationGradGpuKernelMod::kernel_attr_map_ = {
    {kReLU6Grad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &ActivationGradGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &ActivationGradGpuKernelMod::LaunchKernel<half>}}},
    {kTanhGrad,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeComplex128)
         .AddInputAttr(kNumberTypeComplex128)
         .AddOutputAttr(kNumberTypeComplex128),
       &ActivationGradGpuKernelMod::LaunchKernel<utils::Complex<double>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &ActivationGradGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &ActivationGradGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &ActivationGradGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &ActivationGradGpuKernelMod::LaunchKernel<half>}}},
    {kEluGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &ActivationGradGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &ActivationGradGpuKernelMod::LaunchKernel<half>}}},
    {kSigmoidGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &ActivationGradGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &ActivationGradGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &ActivationGradGpuKernelMod::LaunchKernel<double>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &ActivationGradGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex128)
         .AddInputAttr(kNumberTypeComplex128)
         .AddOutputAttr(kNumberTypeComplex128),
       &ActivationGradGpuKernelMod::LaunchKernel<utils::Complex<double>>}}}};

bool ActivationGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();

  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR)
      << "For 'ActivationGrad', the kernel name must be in "
      << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, ActivationGradGpuKernelMod::ActivationGradFunc>>>(
           kernel_attr_map_)
      << ", but got " << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = kernel_attr_map_.at(kernel_name_)[index].second;

  static const std::map<std::string, cudnnActivationMode_t> activation_mode_map = {
    {kReLU6Grad, CUDNN_ACTIVATION_CLIPPED_RELU},
    {kTanhGrad, CUDNN_ACTIVATION_TANH},
    {kEluGrad, CUDNN_ACTIVATION_ELU},
    {kSigmoidGrad, CUDNN_ACTIVATION_SIGMOID}};
  auto mode_iter = activation_mode_map.find(kernel_name_);
  if (mode_iter == activation_mode_map.end()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', only support these activations: "
                  << kernel::Map2Str<std::map, cudnnActivationMode_t>(activation_mode_map) << ", but got "
                  << kernel_name_;
    return KRET_RESIZE_FAILED;
  }
  mode_ = mode_iter->second;

  const auto dtype = inputs.at(kIndex0)->GetDtype();
  if (((dtype == kNumberTypeFloat64) || (dtype == kNumberTypeComplex64) || (dtype == kNumberTypeComplex128)) &&
      (kernel_name_ != kTanhGrad) && (kernel_name_ != kSigmoidGrad)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', only tanh and sigmoid support complex input, but got "
                  << kernel_name_ << " with dtype " << TypeIdLabel(inputs.at(kIndex0)->GetDtype());
  }

  return true;
}

int ActivationGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  size_t input_num = inputs.size();
  if (input_num != 2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num;
    return KRET_RESIZE_FAILED;
  }
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  is_null_input_ = CHECK_NULL_INPUT(input_shape_);
  if (is_null_input_) {
    return KRET_OK;
  }

  const auto dtype = inputs.at(kIndex0)->GetDtype();
  if (((dtype == kNumberTypeFloat64) || (dtype == kNumberTypeComplex64) || (dtype == kNumberTypeComplex128)) &&
      ((kernel_name_ == kTanhGrad) || (kernel_name_ == kSigmoidGrad))) {
    // Does not call Cudnn
    return KRET_OK;
  }

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&data_descriptor_),
                                      "For 'ActivationGrad', cudnnCreateTensorDescriptor failed.");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateActivationDescriptor(&activation_desc_),
                                      "For 'ActivationGrad', cudnnCreateActivationDescriptor failed.");
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs.at(kIndex0)->GetDtype()));
  CheckTensorSize({input_shape_});
  ShapeVector shape;
  double coef = (mode_ == CUDNN_ACTIVATION_CLIPPED_RELU) ? ReLU6_UP_TURNING_POINT : 0.0;
  if (mode_ == CUDNN_ACTIVATION_ELU) coef = 1.0;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetActivationDescriptor(activation_desc_, mode_, CUDNN_PROPAGATE_NAN, coef),
                                      "For 'ActivationGrad', cudnnSetActivationDescriptor failed.");

  const int split_dim = 4;
  if (input_shape_.size() <= split_dim) {
    ShapeNdTo4d(input_shape_, &shape);
    if (inputs.at(kIndex0)->GetFormat() == mindspore::Format::NHWC) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetTensor4dDescriptor(data_descriptor_, CUDNN_TENSOR_NHWC, cudnn_data_type_, LongToInt(shape[0]),
                                   LongToInt(shape[3]), LongToInt(shape[1]), LongToInt(shape[2])),
        "For 'ActivationGrad', cudnnSetTensor4dDescriptor failed.");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetTensor4dDescriptor(data_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, LongToInt(shape[0]),
                                   LongToInt(shape[1]), LongToInt(shape[2]), LongToInt(shape[3])),
        "For 'ActivationGrad', cudnnSetTensor4dDescriptor failed.");
    }
  } else {
    CudnnSetTensorNdDescriptor(input_shape_, data_descriptor_, cudnn_data_type_, kernel_name_);
  }
  return KRET_OK;
}

std::vector<KernelAttr> ActivationGradGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR)
      << "For 'ActivationGrad', the kernel name must be in "
      << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, ActivationGradGpuKernelMod::ActivationGradFunc>>>(
           kernel_attr_map_)
      << ", but got " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ActivationGradFunc> &item) { return item.first; });
  return support_list;
}

template <typename T>
bool ActivationGradGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  T *dy = nullptr;
  T *y = nullptr;
  if (mode_ == CUDNN_ACTIVATION_ELU || mode_ == CUDNN_ACTIVATION_CLIPPED_RELU) {
    dy = GetDeviceAddress<T>(inputs, 0);
    y = GetDeviceAddress<T>(inputs, 1);
  } else {
    y = GetDeviceAddress<T>(inputs, 0);
    dy = GetDeviceAddress<T>(inputs, 1);
  }
  T *dx = GetDeviceAddress<T>(outputs, 0);

  constexpr bool use_unary =
    std::is_same_v<T, double> || std::is_same_v<T, utils::Complex<float>> || std::is_same_v<T, utils::Complex<double>>;
  if constexpr (use_unary) {
    if (kernel_name_ == kTanhGrad) {
      TanhGrad(y, dy, dx, input_size_list_[0] / sizeof(T), reinterpret_cast<cudaStream_t>(cuda_stream_));
    } else {
      SigmoidGrad(y, dy, dx, input_size_list_[0] / sizeof(T), reinterpret_cast<cudaStream_t>(cuda_stream_));
    }
    return true;
  }

  const float alpha = 1;
  const float beta = 0;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnActivationBackward(cudnn_handle_, activation_desc_, &alpha, data_descriptor_, y, data_descriptor_, dy,
                            data_descriptor_, y, &beta, data_descriptor_, dx),
    "For 'ActivationGrad', cudnnActivationBackward failed.");

  return true;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ReLU6Grad,
                                 []() { return std::make_shared<ActivationGradGpuKernelMod>(kReLU6Grad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, TanhGrad,
                                 []() { return std::make_shared<ActivationGradGpuKernelMod>(kTanhGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, EluGrad,
                                 []() { return std::make_shared<ActivationGradGpuKernelMod>(kEluGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SigmoidGrad,
                                 []() { return std::make_shared<ActivationGradGpuKernelMod>(kSigmoidGrad); });
}  // namespace kernel
}  // namespace mindspore
