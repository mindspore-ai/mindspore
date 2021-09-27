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

#include "backend/kernel_compiler/cpu/strided_slice_grad_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "runtime/device/cpu/cpu_device_address.h"
#include "nnacl/fp32_grad/strided_slice_grad.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
void StridedSliceGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  param_ = (struct StridedSliceParameter *)malloc(sizeof(struct StridedSliceParameter));
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "malloc StridedSliceGradParameter failed.";
  }
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  switch (dtype_) {
    case kNumberTypeFloat32:
      param_->data_type = kDataTypeFloat;
      break;
    default:
      MS_LOG(ERROR) << "Not supported data type: " << dtype_;
  }
  std::vector<size_t> input_shape_me = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  (void)std::transform(input_shape_me.begin(), input_shape_me.end(), std::back_inserter(input_shape_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  param_->num_axes_ = input_shape_me.size();
  param_->in_shape_length_ = input_shape_me.size();
  std::vector<int64_t> begin_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, BEGIN);
  (void)std::transform(begin_me.begin(), begin_me.end(), std::back_inserter(begin_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto strides = prim->GetAttr(STRIDES);
  std::vector<int64_t> strides_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
  std::vector<int64_t> end_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, END);
  (void)std::transform(strides_me.begin(), strides_me.end(), std::back_inserter(strides_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  (void)std::transform(end_me.begin(), end_me.end(), std::back_inserter(end_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (strides_.size() != end_.size() || strides_.size() != output_shape_.size()) {
    MS_LOG(EXCEPTION) << "stride|end|input size must be equal";
  }
  ExpandAllMemberDims();
  std::copy(input_shape_.begin(), input_shape_.end(), param_->in_shape_);
  std::copy(begin_.begin(), begin_.end(), param_->begins_);
  std::copy(strides_.begin(), strides_.end(), param_->strides_);
  std::copy(end_.begin(), end_.end(), param_->ends_);
}

void StridedSliceGradCPUKernel::ExpandAllMemberDims() {
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

bool StridedSliceGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> & /* workspace */,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  bool ret{true};
  if (dtype_ == kNumberTypeFloat32) {
    ret = LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "StridedSliceGrad op only support float32";
    return false;
  }
  return ret;
}

template <typename T>
bool StridedSliceGradCPUKernel::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
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
    MS_LOG(ERROR) << "StridedSliceGrad error error_code[" << ret << "]";
    return false;
  }
  return true;
}

StridedSliceGradCPUKernel::~StridedSliceGradCPUKernel() {
  if (param_ != nullptr) {
    free(param_);
    param_ = nullptr;
  }
}
}  // namespace kernel
}  // namespace mindspore
