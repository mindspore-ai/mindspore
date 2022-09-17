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

#include "plugin/device/gpu/kernel/arrays/fill_diagonal_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kFillDiagonalInputNum = 1;
const size_t kFillDiagonalOutputNum = 1;
const size_t kInputDimIndex0 = 0;
const size_t kInputNull = 0;
const size_t kInputDimIndex1 = 1;
const int64_t kInputMinDim = 2;
constexpr int64_t kParallelDataNums = 512 * 1024;
}  // namespace

bool FillDiagonalGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::FillDiagonal>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float32, int32, int64], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);

  fill_value_ = kernel_ptr_->get_fill_value();
  wrap_ = kernel_ptr_->get_wrap();

  return true;
}

int FillDiagonalGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just
    // return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> input_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                          inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  matrix_row_ = input_shape[kInputDimIndex0];
  matrix_col_ = input_shape[kInputDimIndex1];
  int64_t min_size = std::min(matrix_row_, matrix_col_);
  input_elements_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
  if (input_elements_ == kInputNull) {
    is_null_input_ = true;
  }
  input_dims = input_shape.size();
  if (input_dims < kInputMinDim) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'x' should be at least 2-D, but got " << input_dims
                  << "-D.";
    return KRET_RESIZE_FAILED;
  }

  if (input_dims == kInputMinDim) {
    for (int64_t i = (input_dims - 1); i >= 0; i--) {
      step += pow(matrix_col_, i);
    }
  } else {
    int64_t prev_i = input_shape[kInputDimIndex0];
    for (auto i : input_shape) {
      if (i != prev_i) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', all dimension of 'x' must be of equal length.";
      }
    }
    std::vector<int64_t> cumprod(input_dims);
    auto dims = input_shape;
    std::partial_sum(dims.begin(), dims.end() - 1, cumprod.begin(), std::multiplies<int64_t>());
    step = 1 + std::accumulate(cumprod.begin(), cumprod.end(), static_cast<int64_t>(0));
  }
  if (wrap_ || input_dims > kInputMinDim || matrix_row_ < matrix_col_) {
    num_diagonal_elements = ceil(static_cast<double>(input_elements_) / step);
  } else {
    num_diagonal_elements = ceil(static_cast<double>(min_size * min_size) / step);
  }
  size_t input_size = input_elements_ * unit_size_;
  input_size_list_.push_back(input_size);
  output_size_list_.push_back(input_size);
  workspace_size_list_.push_back(sizeof(bool));
  return KRET_OK;
}

template <typename T>
bool FillDiagonalGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output, input, input_elements_ * unit_size_, cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cudaMemcpyAsync output 'output' from 'input' failed.");
  CalFillDiagonal(num_diagonal_elements, fill_value_, step, output, device_id_,
                  reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, FillDiagonalGpuKernelMod::FillDiagonalFunc>> FillDiagonalGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &FillDiagonalGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &FillDiagonalGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &FillDiagonalGpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> FillDiagonalGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FillDiagonalFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, FillDiagonal, FillDiagonalGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
