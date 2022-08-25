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

#include "plugin/device/gpu/kernel/nn/pdist_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr size_t kZeroindex = 0;
constexpr size_t kOneindex = 1;
constexpr size_t kTwoindex = 2;
bool PDistGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::PdistGrad>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float16, float32, double], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  p_ = kernel_ptr_->get_p();
  input_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  return true;
}

int PDistGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }

  ResetResource();
  std::vector<int64_t> y_grad_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> x_shape = inputs[kIndex1]->GetShapeVector();
  std::vector<int64_t> y_shape = inputs[kIndex2]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[kIndex0]->GetShapeVector();
  int64_t y_grad_dim = y_grad_shape.size();
  int64_t x_dim = x_shape.size();
  int64_t y_dim = y_shape.size();
  if (y_grad_dim != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'y_grad' must be 1-D,"
                  << " but got " << y_grad_dim << "-D.";
    return false;
  }
  if (x_dim != kTwoindex) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'x' must be 2-D,"
                  << " but got " << x_dim << "-D.";
    return false;
  }
  if (y_dim != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'y' must be 1-D,"
                  << " but got " << y_dim << "-D.";
    return false;
  }
  y_grad_size_ = std::accumulate(y_grad_shape.begin(), y_grad_shape.end(), 1, std::multiplies<int64_t>());
  x_size_ = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>());
  y_size_ = std::accumulate(y_shape.begin(), y_shape.end(), 1, std::multiplies<int64_t>());
  size_t x_grad_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  matrix_row_ = x_shape[x_shape.size() - kTwoindex];
  matrix_col_ = x_shape[x_shape.size() - kOneindex];
  if (x_grad_size == 0) {
    is_null_input_ = true;
  }
  size_t y_grad_size = y_grad_size_ * input_type_size_;
  size_t x_size = x_size_ * input_type_size_;
  size_t y_size = y_size_ * input_type_size_;
  size_t output_size = x_grad_size * input_type_size_;
  size_t work_size = ((matrix_row_ - 1) * x_size_) * input_type_size_;
  input_size_list_.push_back(y_grad_size);
  input_size_list_.push_back(x_size);
  input_size_list_.push_back(y_size);
  output_size_list_.push_back(output_size);
  workspace_size_list_.push_back(work_size);
  return KRET_OK;
}

template <typename T>
bool PDistGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  T *y_grad = GetDeviceAddress<T>(inputs, kZeroindex);
  T *x = GetDeviceAddress<T>(inputs, kOneindex);
  T *y = GetDeviceAddress<T>(inputs, kTwoindex);
  T *output = GetDeviceAddress<T>(outputs, kZeroindex);
  T *buffer = GetDeviceAddress<T>(workspace, kZeroindex);
  CalPDistGrad(x_size_, y_size_, y_grad_size_, y_grad, x, y, matrix_row_, matrix_col_, p_, output, buffer, device_id_,
               reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, PDistGradGpuKernelMod::PDistGradFunc>> PDistGradGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &PDistGradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &PDistGradGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> PDistGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, PDistGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, PdistGrad, PDistGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
