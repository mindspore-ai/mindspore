/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/fractional_max_pool3d_grad_with_fixed_ksize_cpu_kernel.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/fractional_max_pool3d_grad_with_fixed_ksize.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDimSize4 = 4;
constexpr size_t kDimSize5 = 5;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kFormatNCDHWIndexC = 0;
constexpr size_t kFormatNCDHWIndexD = 1;
constexpr size_t kFormatNCDHWIndexH = 2;
constexpr size_t kFormatNCDHWIndexW = 3;
constexpr size_t kFormatNDHWCIndexD = 0;
constexpr size_t kFormatNDHWCIndexH = 1;
constexpr size_t kFormatNDHWCIndexW = 2;
constexpr size_t kFormatNDHWCIndexC = 3;
constexpr size_t kInputsNum = 3;
constexpr size_t kOutputsNum = 1;

#define ADD_KERNEL(t1, t2, t3, t4) \
  KernelAttr()                     \
    .AddInputAttr(kNumberType##t1) \
    .AddInputAttr(kNumberType##t2) \
    .AddInputAttr(kNumberType##t3) \
    .AddOutputAttr(kNumberType##t4)
}  // namespace

bool FractionalMaxPool3DGradWithFixedKsizeCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                             const std::vector<KernelTensorPtr> &inputs,
                                                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  out_backprop_type_ = inputs[kInputIndex1]->GetDtype();
  argmax_type_ = inputs[kInputIndex2]->GetDtype();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::FractionalMaxPool3DGradWithFixedKsize>(base_operator);
  data_format_ = kernel_ptr->get_data_format();
  return true;
}

int FractionalMaxPool3DGradWithFixedKsizeCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                              const std::vector<KernelTensorPtr> &inputs,
                                                              const std::vector<KernelTensorPtr> &outputs,
                                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[kInputIndex0]->GetDeviceShapeAdaptively();
  out_backprop_shape_ = inputs[kInputIndex1]->GetDeviceShapeAdaptively();
  argmax_shape_ = inputs[kInputIndex2]->GetDeviceShapeAdaptively();
  size_t input_dims = input_shape_.size();
  size_t out_backprop_dims = out_backprop_shape_.size();
  size_t argmax_dims = argmax_shape_.size();
  if (data_format_ == "NCDHW") {
    c_dim_ = kFormatNCDHWIndexC;
    d_dim_ = kFormatNCDHWIndexD;
    h_dim_ = kFormatNCDHWIndexH;
    w_dim_ = kFormatNCDHWIndexW;
  } else {
    c_dim_ = kFormatNDHWCIndexC;
    d_dim_ = kFormatNDHWCIndexD;
    h_dim_ = kFormatNDHWCIndexH;
    w_dim_ = kFormatNDHWCIndexW;
  }
  if (input_shape_.size() == kDimSize5) {
    inputN_ = input_shape_[0];
    c_dim_++;
    d_dim_++;
    h_dim_++;
    w_dim_++;
  }
  inputC_ = input_shape_[c_dim_];
  inputD_ = input_shape_[d_dim_];
  inputH_ = input_shape_[h_dim_];
  inputW_ = input_shape_[w_dim_];
  outputD_ = out_backprop_shape_[d_dim_];
  outputH_ = out_backprop_shape_[h_dim_];
  outputW_ = out_backprop_shape_[w_dim_];
  if (!(input_dims == kDimSize4 || input_dims == kDimSize5)) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "', the dimension of 'input' must be equal to 4 or 5, but got " << input_dims << ".";
  }
  for (size_t i = 0; i < input_dims; i++) {
    if (input_shape_[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', expected 'input' have non-empty spatial dimensions, but 'input' has sizes "
                               << input_shape_[i] << " with dimension " << i << " being empty.";
    }
  }
  if (!(out_backprop_dims == kDimSize4 || out_backprop_dims == kDimSize5)) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "', the dimension of 'out_backprop' must be equal to 4 or 5, but got "
                            << out_backprop_dims << ".";
  }
  for (size_t i = 0; i < out_backprop_dims; i++) {
    if (out_backprop_shape_[i] <= 0) {
      MS_EXCEPTION(ValueError)
        << "For '" << kernel_name_
        << "', expected 'out_backprop' have non-empty spatial dimensions, but 'out_backprop' has sizes "
        << out_backprop_shape_[i] << " with dimension " << i << " being empty.";
    }
  }
  if (!(argmax_dims == kDimSize4 || argmax_dims == kDimSize5)) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "', the dimension of 'argmax' must be equal to 4 or 5, but got " << argmax_dims << ".";
  }
  for (size_t i = 0; i < argmax_dims; i++) {
    if (argmax_shape_[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', expected 'argmax' have non-empty spatial dimensions, but 'argmax' has sizes "
                               << argmax_shape_[i] << " with dimension " << i << " being empty.";
    }
  }

  return KRET_OK;
}

template <typename backprop_t, typename argmax_t>
bool FractionalMaxPool3DGradWithFixedKsizeCPUKernelMod::GradComputeTemplate(const std::vector<AddressPtr> &inputs,
                                                                            const std::vector<AddressPtr> &outputs) {
  auto out_backprop_data = reinterpret_cast<backprop_t *>(inputs[1]->addr);
  auto argmax_data = reinterpret_cast<argmax_t *>(inputs[2]->addr);
  auto output_data = reinterpret_cast<backprop_t *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size;
  if (memset_s(output_data, output_size, 0, output_size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }
  if (input_shape_.size() == kDimSize4) {
    auto shard_fractional_max_pool3d_grad_with_fixed_ksize = [&](size_t start, size_t end) {
      for (auto plane = start; plane < end; ++plane) {
        backprop_t *outputForPlane = output_data + plane * inputD_ * inputH_ * inputW_;
        backprop_t *outbackpropForPlane = out_backprop_data + plane * outputD_ * outputH_ * outputW_;
        argmax_t *argmaxForPlane = argmax_data + plane * outputD_ * outputH_ * outputW_;
        int64_t h, w, t;
        for (t = 0; t < outputD_; ++t) {
          for (h = 0; h < outputH_; ++h) {
            for (w = 0; w < outputW_; ++w) {
              argmax_t outputIndex = t * outputH_ * outputW_ + h * outputW_ + w;
              argmax_t index = argmaxForPlane[outputIndex];
              if (index < 0 && index >= inputD_ * inputH_ * inputW_) {
                MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', index value is illegal.";
              }
              outputForPlane[index] += outbackpropForPlane[outputIndex];
            }
          }
        }
      }
    };
    CPUKernelUtils::ParallelFor(shard_fractional_max_pool3d_grad_with_fixed_ksize, LongToSize(inputC_));
  } else {
    auto shard_fractional_max_pool3d_grad_with_fixed_ksize = [&](size_t start, size_t end) {
      for (auto batch = start; batch < end; ++batch) {
        for (int64_t plane = 0; plane < inputC_; ++plane) {
          auto output_data_n = output_data + batch * inputC_ * inputW_ * inputH_ * inputD_;
          auto out_backprop_data_n = out_backprop_data + batch * inputC_ * outputW_ * outputH_ * outputD_;
          auto argmax_data_n = argmax_data + batch * inputC_ * outputW_ * outputH_ * outputD_;
          backprop_t *outputForPlane = output_data_n + plane * inputD_ * inputH_ * inputW_;
          backprop_t *outbackpropForPlane = out_backprop_data_n + plane * outputD_ * outputH_ * outputW_;
          argmax_t *argmaxForPlane = argmax_data_n + plane * outputD_ * outputH_ * outputW_;
          int64_t h, w, t;
          for (t = 0; t < outputD_; ++t) {
            for (h = 0; h < outputH_; ++h) {
              for (w = 0; w < outputW_; ++w) {
                argmax_t outputIndex = t * outputH_ * outputW_ + h * outputW_ + w;
                argmax_t index = argmaxForPlane[outputIndex];
                if (index < 0 && index >= inputD_ * inputH_ * inputW_) {
                  MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', index value is illegal.";
                }
                outputForPlane[index] += outbackpropForPlane[outputIndex];
              }
            }
          }
        }
      }
    };
    CPUKernelUtils::ParallelFor(shard_fractional_max_pool3d_grad_with_fixed_ksize, LongToSize(inputN_));
  }
  return true;
}

template <typename backprop_t>
bool FractionalMaxPool3DGradWithFixedKsizeCPUKernelMod::DoComputeWithArgmaxType(const std::vector<AddressPtr> &inputs,
                                                                                const std::vector<AddressPtr> &outputs,
                                                                                TypeId argmax_type) {
  switch (argmax_type) {
    case kNumberTypeInt32:
      return GradComputeTemplate<backprop_t, int32_t>(inputs, outputs);
    case kNumberTypeInt64:
      return GradComputeTemplate<backprop_t, int64_t>(inputs, outputs);
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the type of 'argmax'" << argmax_type
                              << "not support, must be in [{DT_INT32, DT_INT64}].";
  }
  return false;
}

bool FractionalMaxPool3DGradWithFixedKsizeCPUKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                                               const std::vector<AddressPtr> &workspace,
                                                               const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  switch (out_backprop_type_) {
    case kNumberTypeFloat16:
      return DoComputeWithArgmaxType<float16>(inputs, outputs, argmax_type_);
    case kNumberTypeFloat32:
      return DoComputeWithArgmaxType<float>(inputs, outputs, argmax_type_);
    case kNumberTypeFloat64:
      return DoComputeWithArgmaxType<double>(inputs, outputs, argmax_type_);
    case kNumberTypeInt32:
      return DoComputeWithArgmaxType<int32_t>(inputs, outputs, argmax_type_);
    case kNumberTypeInt64:
      return DoComputeWithArgmaxType<int64_t>(inputs, outputs, argmax_type_);
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the type of 'out_backprop'" << out_backprop_type_
                              << "not support, must be in [{DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}].";
  }
  return true;
}

std::vector<KernelAttr> FractionalMaxPool3DGradWithFixedKsizeCPUKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Int32, Int32, Int32, Int32),       ADD_KERNEL(Int64, Int64, Int32, Int64),
    ADD_KERNEL(Float16, Float16, Int32, Float16), ADD_KERNEL(Float32, Float32, Int32, Float32),
    ADD_KERNEL(Float64, Float64, Int32, Float64), ADD_KERNEL(Int32, Int32, Int64, Int32),
    ADD_KERNEL(Int64, Int64, Int64, Int64),       ADD_KERNEL(Float16, Float16, Int64, Float16),
    ADD_KERNEL(Float32, Float32, Int64, Float32), ADD_KERNEL(Float64, Float64, Int64, Float64)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FractionalMaxPool3DGradWithFixedKsize,
                      FractionalMaxPool3DGradWithFixedKsizeCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
