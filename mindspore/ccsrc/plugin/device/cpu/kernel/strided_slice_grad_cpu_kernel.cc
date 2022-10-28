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

#include "plugin/device/cpu/kernel/strided_slice_grad_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "ops/grad/strided_slice_grad.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "nnacl/fp32_grad/strided_slice_grad.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
bool StridedSliceGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  dtype_ = inputs.at(kIndex0)->GetDtype();
  param_ = (struct StridedSliceParameter *)malloc(sizeof(struct StridedSliceParameter));
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc StridedSliceGradParameter failed.";
  }
  switch (dtype_) {
    case kNumberTypeFloat32:
      param_->data_type = ::kNumberTypeFloat32;
      break;
    default:
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dtype of input must be float32, but got " << dtype_;
  }
  return true;
}
int StridedSliceGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  input_shape_ = inputs.at(kIndex0)->GetDeviceShapeAdaptively();
  param_->num_axes_ = SizeToInt(input_shape_.size());
  param_->in_shape_length_ = SizeToInt(input_shape_.size());
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  begin_.clear();
  strides_.clear();
  end_.clear();
  std::vector<int64_t> begin_me = GetValue<std::vector<int64_t>>(prim->GetAttr(BEGIN));
  std::vector<int64_t> strides_me = GetValue<std::vector<int64_t>>(prim->GetAttr(STRIDES));
  std::vector<int64_t> end_me = GetValue<std::vector<int64_t>>(prim->GetAttr(END));
  (void)std::transform(begin_me.begin(), begin_me.end(), std::back_inserter(begin_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  (void)std::transform(strides_me.begin(), strides_me.end(), std::back_inserter(strides_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  (void)std::transform(end_me.begin(), end_me.end(), std::back_inserter(end_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (strides_.size() != end_.size() || strides_.size() != output_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the length of 'stride' and 'end' must be equal "
                         "to the dimension of output, but got the length of 'stride': "
                      << strides_.size() << ", the length of 'end': " << end_.size()
                      << ", the dimension of output: " << output_shape_.size();
  }
  ExpandAllMemberDims();
  std::copy(input_shape_.begin(), input_shape_.end(), param_->in_shape_);
  std::copy(begin_.begin(), begin_.end(), param_->begins_);
  std::copy(strides_.begin(), strides_.end(), param_->strides_);
  std::copy(end_.begin(), end_.end(), param_->ends_);
  return KRET_OK;
}

void StridedSliceGradCpuKernelMod::ExpandAllMemberDims() {
  auto input_len = input_shape_.size();
  if (input_len < DIMENSION_8D) {
    for (size_t i = 0; i < DIMENSION_8D - input_len; ++i) {
      input_shape_.insert(input_shape_.begin(), 1);
    }
  }
  auto output_len = output_shape_.size();
  if (output_len < DIMENSION_8D) {
    for (size_t i = 0; i < DIMENSION_8D - output_len; ++i) {
      output_shape_.insert(output_shape_.begin(), 1);
      begin_.insert(begin_.begin(), 0);
      strides_.insert(strides_.begin(), 1);
      end_.insert(end_.begin(), 1);
    }
  }
  param_->num_axes_ = DIMENSION_8D;
  param_->in_shape_length_ = DIMENSION_8D;

  for (size_t i = 0; i < DIMENSION_8D; ++i) {
    if (begin_[i] < 0) {
      begin_[i] += input_shape_[i];
    }
    if (end_[i] < 0) {
      end_[i] += input_shape_[i];
    }
  }
}

bool StridedSliceGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> & /* workspace */,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  bool ret{true};
  if (dtype_ == kNumberTypeFloat32) {
    ret = LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dtype of input must be float32, but got " << dtype_;
    return false;
  }
  return ret;
}

template <typename T>
bool StridedSliceGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  auto dx = reinterpret_cast<float *>(output_addr);
  auto dy = reinterpret_cast<float *>(input_addr);

  auto ElementsNum = std::accumulate(output_shape_.begin(), output_shape_.end(), 1LL, std::multiplies<int>());
  std::fill(dx, dx + ElementsNum, 0.f);
  std::vector<int> output_;
  (void)std::transform(output_shape_.begin(), output_shape_.end(), std::back_inserter(output_),
                       [](const size_t &value) { return static_cast<int>(value); });
  auto ret = DoStridedSliceGrad(dy, dx, output_.data(), param_);
  if (ret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', raise error, error no: " << ret;
    return false;
  }
  return true;
}

StridedSliceGradCpuKernelMod::~StridedSliceGradCpuKernelMod() {
  if (param_ != nullptr) {
    free(param_);
    param_ = nullptr;
  }
}
}  // namespace kernel
}  // namespace mindspore
