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
#include "plugin/device/cpu/kernel/fractional_max_pool_grad_with_fixed_ksize_cpu_kernel.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kInputsNum = 3;
const size_t kOutputsNum = 1;
const size_t kInputIndex0 = 0;
const size_t kInputIndex1 = 1;
const size_t kInputIndex2 = 2;
const size_t kInputsDimSize = 4;
const size_t kInputsDimIndexN = 0;
const size_t kInputsDimIndexC = 1;
const size_t kInputsDimIndexH = 2;
const size_t kInputsDimIndexW = 3;

#define ADD_KERNEL(t1, t2, t3, t4) \
  KernelAttr()                     \
    .AddInputAttr(kNumberType##t1) \
    .AddInputAttr(kNumberType##t2) \
    .AddInputAttr(kNumberType##t3) \
    .AddOutputAttr(kNumberType##t4)
}  // namespace

bool FractionalMaxPoolGradWithFixedKsizeCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                           const std::vector<KernelTensorPtr> &inputs,
                                                           const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  constexpr size_t input_num = kInputsNum;
  constexpr size_t output_num = kOutputsNum;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  out_backprop_type_ = inputs[kInputIndex1]->GetDtype();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::FractionalMaxPoolGradWithFixedKsize>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  data_format_ = kernel_ptr->get_data_format();
  if (data_format_ != "NCHW") {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the attr data_format must be NCHW.";
  }
  return true;
}

int FractionalMaxPoolGradWithFixedKsizeCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                            const std::vector<KernelTensorPtr> &inputs,
                                                            const std::vector<KernelTensorPtr> &outputs,
                                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kInputIndex0]->GetDeviceShapeAdaptively();
  out_backprop_shape_ = inputs[kInputIndex1]->GetDeviceShapeAdaptively();
  argmax_shape_ = inputs[kInputIndex2]->GetDeviceShapeAdaptively();
  if (input_shape_.size() != kInputsDimSize) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', The dim of input origin_input must be 4, but got "
                             << input_shape_.size() << ".";
  }
  input_n_ = input_shape_[kInputsDimIndexN];
  input_c_ = input_shape_[kInputsDimIndexC];
  input_h_ = input_shape_[kInputsDimIndexH];
  input_w_ = input_shape_[kInputsDimIndexW];
  if (out_backprop_shape_.size() != kInputsDimSize) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', The dim of input out_backprop must be 4, but got "
                             << out_backprop_shape_.size() << ".";
  }
  out_backprop_h_ = out_backprop_shape_[kInputsDimIndexH];
  out_backprop_w_ = out_backprop_shape_[kInputsDimIndexW];
  if (argmax_shape_.size() != kInputsDimSize) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', The dim of input argmax must be 4, but got "
                             << argmax_shape_.size() << ".";
  }
  for (size_t i = 0; i < kInputsDimSize; i++) {
    if (out_backprop_shape_[i] != argmax_shape_[i]) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', The shape of input out_backprop and input argmax must be equal.";
    }
  }

  if (input_n_ != out_backprop_shape_[kInputsDimIndexN]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', The first dimension of three inputs must be equal.";
  }
  if (input_c_ != out_backprop_shape_[kInputsDimIndexC]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', The second dimension of three inputs must be equal.";
  }
  return ret;
}

bool FractionalMaxPoolGradWithFixedKsizeCPUKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                                             const std::vector<AddressPtr> &workspace,
                                                             const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  switch (out_backprop_type_) {
    case kNumberTypeFloat16:
      return GradComputeTemplate<float16>(inputs, outputs);
    case kNumberTypeFloat32:
      return GradComputeTemplate<float>(inputs, outputs);
    case kNumberTypeFloat64:
      return GradComputeTemplate<double>(inputs, outputs);
    case kNumberTypeInt32:
      return GradComputeTemplate<int32_t>(inputs, outputs);
    case kNumberTypeInt64:
      return GradComputeTemplate<int64_t>(inputs, outputs);
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', out_backprop_type" << out_backprop_type_
                              << "not support, must be in [{DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}].";
  }
  return true;
}

template <typename backprop_t>
bool FractionalMaxPoolGradWithFixedKsizeCPUKernelMod::GradComputeTemplate(
  const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) const {
  backprop_t *out_backprop_ptr = static_cast<backprop_t *>(inputs[1]->addr);
  int64_t *argmax_ptr = static_cast<int64_t *>(inputs[2]->addr);
  backprop_t *output_ptr = static_cast<backprop_t *>(outputs[0]->addr);

  auto shard_fractional_max_pool_grad_with_fixed_ksize = [&](size_t start, size_t end) {
    for (size_t n = start; n < end; n++) {
      backprop_t *out_backpropForPlane = out_backprop_ptr + n * input_c_ * out_backprop_h_ * out_backprop_w_;
      int64_t *argmaxForPlane = argmax_ptr + n * input_c_ * out_backprop_h_ * out_backprop_w_;
      backprop_t *outputForPlane = output_ptr + n * input_c_ * input_h_ * input_w_;

      FractionalMaxPoolGradWithFixedKsizeCompute<backprop_t>(out_backpropForPlane, argmaxForPlane, outputForPlane);
    }
  };
  CPUKernelUtils::ParallelFor(shard_fractional_max_pool_grad_with_fixed_ksize, LongToSize(input_n_));
  return true;
}

template <typename backprop_t>
void FractionalMaxPoolGradWithFixedKsizeCPUKernelMod::FractionalMaxPoolGradWithFixedKsizeCompute(
  backprop_t *out_backpropForPlane, int64_t *argmaxForPlane, backprop_t *outputForPlane) const {
  for (int64_t plane = 0; plane < input_c_; plane++) {
    backprop_t *out_backpropPlane = out_backpropForPlane + plane * out_backprop_h_ * out_backprop_w_;
    int64_t *argmaxPlane = argmaxForPlane + plane * out_backprop_h_ * out_backprop_w_;
    backprop_t *outputPlane = outputForPlane + plane * input_h_ * input_w_;

    for (int64_t i = 0; i < input_h_; i++) {
      for (int64_t j = 0; j < input_w_; j++) {
        outputPlane[i * input_w_ + j] = static_cast<backprop_t>(0);
      }
    }

    for (int64_t h = 0; h < out_backprop_h_; h++) {
      for (int64_t w = 0; w < out_backprop_w_; w++) {
        int input_index = h * out_backprop_w_ + w;
        if (input_index < 0 || input_index >= (out_backprop_h_ * out_backprop_w_)) {
          MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the index value of argmax is illegal.";
        }
        int output_index = argmaxPlane[input_index];
        if (output_index < 0 || output_index >= (input_h_ * input_w_)) {
          MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the index value of output is illegal.";
        }
        outputPlane[output_index] += out_backpropPlane[input_index];
      }
    }
  }
}

std::vector<KernelAttr> FractionalMaxPoolGradWithFixedKsizeCPUKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Int32, Float16, Int64, Float16), ADD_KERNEL(Int32, Float32, Int64, Float32),
    ADD_KERNEL(Int32, Float64, Int64, Float64), ADD_KERNEL(Int32, Int32, Int64, Int32),
    ADD_KERNEL(Int32, Int64, Int64, Int64),     ADD_KERNEL(Int64, Float16, Int64, Float16),
    ADD_KERNEL(Int64, Float32, Int64, Float32), ADD_KERNEL(Int64, Float64, Int64, Float64),
    ADD_KERNEL(Int64, Int32, Int64, Int32),     ADD_KERNEL(Int64, Int64, Int64, Int64)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FractionalMaxPoolGradWithFixedKsize,
                      FractionalMaxPoolGradWithFixedKsizeCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
