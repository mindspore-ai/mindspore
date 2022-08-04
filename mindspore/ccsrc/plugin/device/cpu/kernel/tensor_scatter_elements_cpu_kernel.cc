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

#include "plugin/device/cpu/kernel/tensor_scatter_elements_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/tensor_scatter_elements.h"

namespace mindspore::kernel {
constexpr size_t kTensorScatterElementsInputsNum = 3;
constexpr size_t kTensorScatterElementsOutputsNum = 1;

namespace {
template <class T>
struct ReductionAdd {
  void operator()(T *a, const T &b) const { (*a) += b; }
};

template <class T>
struct ReductionAssignment {
  void operator()(T *a, const T &b) const { (*a) = b; }
};
}  // namespace

int TensorScatterElementsCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, KRET_RESIZE_FAILED);
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  kernel_name_ = base_operator->name();
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  input_dims_ = input_shape.size();
  if (input_dims_ < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input_x' should be greater than or equal to 1, but got " << input_dims_
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  indices_shape_ = inputs[kIndex1]->GetShapeVector();
  auto update_shape = inputs[kIndex2]->GetShapeVector();
  if (indices_shape_ != update_shape) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'indice' and the shape of 'update' should be same, but got "
                  << "indice shape: " << indices_shape_ << "; "
                  << "update shape: " << update_shape << ".";
    return KRET_RESIZE_FAILED;
  }
  if (input_dims_ != indices_shape_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input_x', 'indice' and 'update' should be same, but got "
                  << "input_x dims: " << input_dims_ << "; "
                  << "indice dims: " << indices_shape_.size() << "; "
                  << "update dims: " << update_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }

  if (base_operator->HasAttr(kAttrAxis)) {
    axis_ = GetValue<int64_t>(base_operator->GetAttr(kAttrAxis));
    if (axis_ < 0) {
      axis_ += static_cast<int64_t>(input_dims_);
    }
  }

  if (axis_ >= static_cast<int64_t>(input_dims_) || axis_ < 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the 'axis' should be less than input dims and greater than or equal 0, but got " << axis_
                  << ", while input dims is: " << input_dims_;
    return KRET_RESIZE_FAILED;
  }

  for (size_t i = 0; i < input_dims_; ++i) {
    if (axis_ != static_cast<int64_t>(i) && input_shape[i] < indices_shape_[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the indices dims should be less than input dims, but got indices dim is: "
                    << indices_shape_[i] << " at axis: " << i << ", while input dim is:" << input_shape[i];
      return KRET_RESIZE_FAILED;
    }
  }

  input_axis_size_ = SizeToInt(input_shape[axis_]);
  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  indices_total_num_ =
    std::accumulate(indices_shape_.begin(), indices_shape_.end(), size_t(1), std::multiplies<size_t>());

  // calculate indices_stride
  indices_stride_.resize(input_dims_, 1);
  for (size_t i = input_dims_ - 1; i > 0; --i) {
    indices_stride_[i - 1] = indices_stride_[i] * static_cast<size_t>(indices_shape_[i]);
  }

  // calculate output_stride
  output_stride_.resize(input_dims_, 1);
  for (size_t i = input_dims_ - 1; i > 0; --i) {
    output_stride_[i - 1] = output_stride_[i] * static_cast<size_t>(input_shape[i]);
  }
  return KRET_OK;
}

template <typename T, typename S, typename ReductionT>
bool TensorScatterElementsCpuKernelMod::Scatter(const ReductionT &reduction_func, T *output, const S *indices,
                                                const T *updates) {
  auto task = [reduction_func, output, indices, updates, this](size_t start, size_t end) {
    for (size_t index = start; index < end; index++) {
      int remain = index;
      int output_offset = 0;
      for (size_t i = 0; i < this->input_dims_; ++i) {
        int output_dim_index = remain / this->indices_stride_[i];
        remain %= this->indices_stride_[i];
        if (i == static_cast<size_t>(this->axis_)) {
          output_dim_index = *(indices + index);
          if (output_dim_index >= this->input_axis_size_ || output_dim_index < -this->input_axis_size_) {
            return;
          }
          if (output_dim_index < 0) {
            output_dim_index += this->input_axis_size_;
          }
        }
        output_offset += this->output_stride_[i] * output_dim_index;
      }
      reduction_func(output + output_offset, *(updates + index));
    }
  };
  ParallelLaunchAutoSearch(task, indices_total_num_, this, &parallel_search_info_, pool_);
  return true;
}

template <typename T, typename S>
bool TensorScatterElementsCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTensorScatterElementsInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTensorScatterElementsOutputsNum, kernel_name_);
  auto *input = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *indices = reinterpret_cast<S *>(inputs[kIndex1]->addr);
  auto *updates = reinterpret_cast<T *>(inputs[kIndex2]->addr);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  auto buffer_size = outputs[kIndex0]->size;
  auto ret = memcpy_s(output, buffer_size, input, input_size_ * sizeof(T));
  if (ret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', memory copy failed. Error no: " << ret;
    return false;
  }
  switch (reduction_type_) {
    case REDUCTION_ASSIGNMENT:
      return Scatter(ReductionAssignment<T>(), output, indices, updates);
    case REDUCTION_ADD:
      return Scatter(ReductionAdd<T>(), output, indices, updates);
    default:
      return false;
  }
}

#define TENSOR_SCATTER_ELEMENTS_CPU_REG(MS_T, MS_S, T, S)                                    \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddInputAttr(MS_T).AddOutputAttr(MS_T), \
    &TensorScatterElementsCpuKernelMod::LaunchKernel<T, S>

const std::vector<std::pair<KernelAttr, TensorScatterElementsCpuKernelMod::KernelRunFunc>>
  &TensorScatterElementsCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, TensorScatterElementsCpuKernelMod::KernelRunFunc>> func_list = {
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeBool, kNumberTypeInt32, bool, int32_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t, int32_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int32_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t, int32_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat16, kNumberTypeInt32, float16, int32_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float, int32_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double, int32_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt64, kNumberTypeInt32, int64_t, int32_t)},

    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat16, kNumberTypeInt64, float16, int64_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
    {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)}};
  return func_list;
}

bool TensorScatterElementsCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::TensorScatterElements>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  kernel_name_ = base_operator->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(ERROR) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
    return false;
  }

  std::string reduction = kernel_ptr->get_reduction();
  if (reduction == "none") {
    reduction_type_ = REDUCTION_ASSIGNMENT;
  } else if (reduction == "add") {
    reduction_type_ = REDUCTION_ADD;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', reduction_type: " << reduction_type_ << " not support now.";
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, TensorScatterElements, []() {
  return std::make_shared<TensorScatterElementsCpuKernelMod>(kTensorScatterElements);
});
}  // namespace mindspore::kernel
