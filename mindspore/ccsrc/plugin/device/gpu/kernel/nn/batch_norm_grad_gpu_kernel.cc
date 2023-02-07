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

#include "plugin/device/gpu/kernel/nn/batch_norm_grad_gpu_kernel.h"
#include <map>
#include <algorithm>
#include <utility>
#include <memory>
#include "mindspore/core/ops/grad/batch_norm_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBatchNormGradInputShapeMaxSize = 4;
constexpr size_t kBatchNormGradInputShapeMinSize = 2;
}  // namespace
bool BatchNormGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BatchNormGrad>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast BatchNormGrad failed!";
    return false;
  }
  if (kernel_name_ == kBatchNormGradOpName) {
    bn_ops_ = CUDNN_BATCHNORM_OPS_BN;
  } else if (kernel_name_ == kBatchNormGradWithActivation) {
    bn_ops_ = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
  } else if (kernel_name_ == kBatchNormGradWithAddAndActivation) {
    bn_ops_ = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
  } else {
    MS_LOG(EXCEPTION) << "Only support these kernel names: " << kBatchNormGradOpName << ", "
                      << kBatchNormGradWithActivation << ", " << kBatchNormGradWithAddAndActivation << ", but got "
                      << kernel_name_;
  }

  InitResource();
  epsilon_ = kernel_ptr->get_epsilon();
  is_train_ = kernel_ptr->get_is_training();
  format_ = kernel_ptr->get_format();
  beta_data_diff_ = kernel_ptr->get_inplace_algo() == "cover" ? 0 : 1;

  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[kIndex0]->GetDtype()));
  size_t input_num = inputs.size();
  if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN) {
    if (input_num != CUDNN_BATCHNORM_OPS_BN_INPUT_NUM) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be "
                        << CUDNN_BATCHNORM_OPS_BN_INPUT_NUM << ", but got " << input_num;
    }
  } else {
    if (input_num != NO_CUDNN_BATCHNORM_OPS_BN_INPUT_NUM) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be "
                        << NO_CUDNN_BATCHNORM_OPS_BN_INPUT_NUM << ", but got " << input_num;
    }
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = kernel_attr_map_.at(kernel_name_)[index].second;

  return true;
}

int BatchNormGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  beta_data_diff_ = 0;

  auto x_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  const size_t x_shape_size = x_shape.size();

  auto format = inputs[kIndex0]->GetFormat();
  if (x_shape_size == kBatchNormGradInputShapeMinSize) {
    format = Format::NCHW;
  } else if (format_ == Format::NHWC) {
    format = Format::NHWC;
  }

  (void)x_shape.insert(x_shape.begin() + (format == Format::NHWC ? kIndex1 : x_shape_size),
                       kBatchNormGradInputShapeMaxSize - x_shape_size, 1);

  is_null_input_ = CHECK_SHAPE_NULL(x_shape, kernel_name_, "input");
  if (is_null_input_) {
    InitSizeLists();
    return true;
  }

  if (x_shape_size == kBatchNormGradInputShapeMinSize) {
    mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else {
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  }

  CheckTensorSize({x_shape});
  SetTensorDescriptor(format, x_shape);
  InitSizeLists();
  return KRET_OK;
}

void BatchNormGradGpuKernelMod::InitResource() {
  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
  if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&y_desc_), "Create y desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateActivationDescriptor(&activation_desc_),
                                        "Create activation descriptor failed");
  }
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dy_desc_), "Create dy desc failed");

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dx_desc_), "Create dx desc failed");
  if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dz_desc_), "Create dz desc failed");
  }
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&scale_bias_diff_desc_), "Create para desc failed");
}

void BatchNormGradGpuKernelMod::InitSizeLists() {
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(x_desc_, &x_size_), "Get x size failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(scale_bias_diff_desc_, &para_size_),
                                      "Get para size failed");

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
                                        handle_, mode_, bn_ops_, x_desc_, y_desc_, dy_desc_, dz_desc_, dx_desc_,
                                        scale_bias_diff_desc_, activation_desc_, &workspace_size_),
                                      "cudnnGetBatchNormalizationBackwardExWorkspaceSize failed");
  if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
    input_size_list_.push_back(para_size_);
    input_size_list_.push_back(x_size_);
  }

  workspace_size_list_.push_back(workspace_size_);
}

void BatchNormGradGpuKernelMod::DestroyResource() noexcept {
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
  if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(y_desc_), "Destroy y desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyActivationDescriptor(activation_desc_),
                                       "Destroy activation descriptor failed");
  }
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dy_desc_), "Destroy dy desc failed");

  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dx_desc_), "Destroy dx desc failed");
  if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dz_desc_), "Destroy z desc failed");
  }
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(scale_bias_diff_desc_), "Destroy para desc failed");
}

template <typename T>
bool BatchNormGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  VARIABLE_NOT_USED(workspace);
  VARIABLE_NOT_USED(stream_ptr);
  if (is_null_input_) {
    return true;
  }
  auto dy = GetDeviceAddress<T>(inputs, kIndex0);
  auto x = GetDeviceAddress<T>(inputs, kIndex1);
  auto scale = GetDeviceAddress<float>(inputs, kIndex2);
  auto save_mean = GetDeviceAddress<float>(inputs, kIndex3);
  auto save_variance = GetDeviceAddress<float>(inputs, kIndex4);
  void *bias = nullptr;
  T *y = nullptr;
  if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
    bias = GetDeviceAddress<float>(inputs, kIndex6);
    y = GetDeviceAddress<T>(inputs, kIndex7);
  }

  auto dx = GetDeviceAddress<T>(outputs, kIndex0);
  auto dscale = GetDeviceAddress<float>(outputs, kIndex1);
  auto dbias = GetDeviceAddress<float>(outputs, kIndex2);
  T *dz = nullptr;
  if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
    dz = GetDeviceAddress<T>(outputs, kIndex3);
  }

  if (is_train_) {
    auto reserve_addr = GetPossiblyNullDeviceAddress<float>(inputs, kIndex5);
    reserve_size_ = inputs[kIndex5]->size;
    void *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, kIndex0);

    const float alpha_data_diff = 1;
    const float alpha_param_diff = 1;
    const float beta_param_diff = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnBatchNormalizationBackwardEx(handle_, mode_, bn_ops_, &alpha_data_diff, &beta_data_diff_, &alpha_param_diff,
                                        &beta_param_diff, x_desc_, x, y_desc_, y, dy_desc_, dy, dz_desc_, dz, dx_desc_,
                                        dx, scale_bias_diff_desc_, scale, bias, dscale, dbias, epsilon_, save_mean,
                                        save_variance, activation_desc_, workspace_addr, workspace_size_, reserve_addr,
                                        reserve_size_),
      "Kernel launch failed");
  } else {
    CalBatchNormGrad(x, dy, scale, save_mean, save_variance, dx, dscale, dbias, epsilon_, batch_, channel_, height_,
                     width_, reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  return true;
}

void BatchNormGradGpuKernelMod::SetTensorDescriptor(const Format &format, const ShapeVector &shape) {
  cudnnTensorFormat_t cudnn_format;
  if (format == Format::NHWC) {
    batch_ = LongToInt(shape[kIndex0]);
    height_ = LongToInt(shape[kIndex1]);
    width_ = LongToInt(shape[kIndex2]);
    channel_ = LongToInt(shape[kIndex3]);
    cudnn_format = CUDNN_TENSOR_NHWC;
  } else {
    batch_ = LongToInt(shape[kIndex0]);
    channel_ = LongToInt(shape[kIndex1]);
    height_ = LongToInt(shape[kIndex2]);
    width_ = LongToInt(shape[kIndex3]);
    cudnn_format = CUDNN_TENSOR_NCHW;
  }

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(x_desc_, cudnn_format, cudnn_data_type_, batch_, channel_, height_, width_),
    "Set x desc failed");

  if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(y_desc_, cudnn_format, cudnn_data_type_, batch_, channel_, height_, width_),
      "Set z desc failed");
  }

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(dy_desc_, cudnn_format, cudnn_data_type_, batch_, channel_, height_, width_),
    "Set dy desc failed");

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(dx_desc_, cudnn_format, cudnn_data_type_, batch_, channel_, height_, width_),
    "Set dx desc failed");

  if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(dz_desc_, cudnn_format, cudnn_data_type_, batch_, channel_, height_, width_),
      "Set z desc failed");
  }

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(scale_bias_diff_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel_, 1, 1),
    "Set para desc failed");

  if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetActivationDescriptor(activation_desc_, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0),
      "cudnnSetActivationDescriptor failed");
  }
}

std::map<std::string, std::vector<std::pair<KernelAttr, BatchNormGradGpuKernelMod::BatchNormGradFunc>>>
  BatchNormGradGpuKernelMod::kernel_attr_map_ = {{kBatchNormGradOpName,
                                                  {{KernelAttr()
                                                      .AddInputAttr(kNumberTypeFloat32)    // dy
                                                      .AddInputAttr(kNumberTypeFloat32)    // x
                                                      .AddInputAttr(kNumberTypeFloat32)    // scale
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_mean
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_variance
                                                      .AddInputAttr(kNumberTypeFloat32)    // reserve
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dx
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dscale
                                                      .AddOutputAttr(kNumberTypeFloat32),  // dbias
                                                    &BatchNormGradGpuKernelMod::LaunchKernel<float>},
                                                   {KernelAttr()
                                                      .AddInputAttr(kNumberTypeFloat16)    // dy
                                                      .AddInputAttr(kNumberTypeFloat16)    // x
                                                      .AddInputAttr(kNumberTypeFloat32)    // scale
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_mean
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_variance
                                                      .AddInputAttr(kNumberTypeFloat32)    // reserve
                                                      .AddOutputAttr(kNumberTypeFloat16)   // dx
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dscale
                                                      .AddOutputAttr(kNumberTypeFloat32),  // dbias
                                                    &BatchNormGradGpuKernelMod::LaunchKernel<half>}}},
                                                 {kBatchNormGradWithActivation,
                                                  {{KernelAttr()
                                                      .AddInputAttr(kNumberTypeFloat32)    // dy
                                                      .AddInputAttr(kNumberTypeFloat32)    // x
                                                      .AddInputAttr(kNumberTypeFloat32)    // scale
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_mean
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_variance
                                                      .AddInputAttr(kNumberTypeFloat32)    // reserve
                                                      .AddInputAttr(kNumberTypeFloat32)    // b
                                                      .AddInputAttr(kNumberTypeFloat32)    // y
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dx
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dscale
                                                      .AddOutputAttr(kNumberTypeFloat32),  // dbias
                                                    &BatchNormGradGpuKernelMod::LaunchKernel<float>},
                                                   {KernelAttr()
                                                      .AddInputAttr(kNumberTypeFloat16)    // dy
                                                      .AddInputAttr(kNumberTypeFloat16)    // x
                                                      .AddInputAttr(kNumberTypeFloat32)    // scale
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_mean
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_variance
                                                      .AddInputAttr(kNumberTypeFloat32)    // reserve
                                                      .AddInputAttr(kNumberTypeFloat32)    // b
                                                      .AddInputAttr(kNumberTypeFloat16)    // y
                                                      .AddOutputAttr(kNumberTypeFloat16)   // dx
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dscale
                                                      .AddOutputAttr(kNumberTypeFloat32),  // dbias
                                                    &BatchNormGradGpuKernelMod::LaunchKernel<half>}}},
                                                 {kBatchNormGradWithAddAndActivation,
                                                  {{KernelAttr()
                                                      .AddInputAttr(kNumberTypeFloat32)    // dy
                                                      .AddInputAttr(kNumberTypeFloat32)    // x
                                                      .AddInputAttr(kNumberTypeFloat32)    // scale
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_mean
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_variance
                                                      .AddInputAttr(kNumberTypeFloat32)    // reserve
                                                      .AddInputAttr(kNumberTypeFloat32)    // b
                                                      .AddInputAttr(kNumberTypeFloat32)    // y
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dx
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dscale
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dbias
                                                      .AddOutputAttr(kNumberTypeFloat32),  // dz
                                                    &BatchNormGradGpuKernelMod::LaunchKernel<float>},
                                                   {KernelAttr()
                                                      .AddInputAttr(kNumberTypeFloat16)    // dy
                                                      .AddInputAttr(kNumberTypeFloat16)    // x
                                                      .AddInputAttr(kNumberTypeFloat32)    // scale
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_mean
                                                      .AddInputAttr(kNumberTypeFloat32)    // save_variance
                                                      .AddInputAttr(kNumberTypeFloat32)    // reserve
                                                      .AddInputAttr(kNumberTypeFloat32)    // b
                                                      .AddInputAttr(kNumberTypeFloat16)    // y
                                                      .AddOutputAttr(kNumberTypeFloat16)   // dx
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dscale
                                                      .AddOutputAttr(kNumberTypeFloat32)   // dbias
                                                      .AddOutputAttr(kNumberTypeFloat16),  // dz
                                                    &BatchNormGradGpuKernelMod::LaunchKernel<half>}}}};

std::vector<KernelAttr> BatchNormGradGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR)
      << "For 'BatchNormGrad', the kernel name must be in "
      << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, BatchNormGradGpuKernelMod::BatchNormGradFunc>>>(
           kernel_attr_map_)
      << ", but got " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BatchNormGradFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BatchNormGrad,
                                 []() { return std::make_shared<BatchNormGradGpuKernelMod>(kBatchNormGradOpName); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BatchNormGradWithActivation, []() {
  return std::make_shared<BatchNormGradGpuKernelMod>(kBatchNormGradWithActivation);
});
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BatchNormGradWithAddAndActivation, []() {
  return std::make_shared<BatchNormGradGpuKernelMod>(kBatchNormGradWithAddAndActivation);
});
}  // namespace kernel
}  // namespace mindspore
