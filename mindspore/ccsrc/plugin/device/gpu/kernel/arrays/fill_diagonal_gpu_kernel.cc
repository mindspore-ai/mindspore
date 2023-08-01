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
constexpr size_t kDiagonalInputsNum = 1;
constexpr size_t kDiagonalOutputsNum = 1;
constexpr size_t kInputDimIndex0 = 0;
constexpr size_t kInputNull = 0;
constexpr size_t kInputDimIndex1 = 1;
constexpr int64_t kInputMinDim = 2;
}  // namespace

bool FillDiagonalGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDiagonalInputsNum, kernel_name_);
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::FillDiagonal>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr_, false);
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
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);

  fill_value_ = kernel_ptr_->get_fill_value();
  wrap_ = kernel_ptr_->get_wrap();

  if (IsOneOfUnsignedType(inputs.at(0)->GetDtype()) && fill_value_ < 0) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", [file_value] should be non_negative for input of unsigned type.";
    return false;
  }

  return true;
}

int FillDiagonalGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDiagonalInputsNum, kernel_name_);
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
  input_dims_ = input_shape.size();
  if (input_dims_ == kInputMinDim) {
    for (int64_t i = (input_dims_ - 1); i >= 0; i--) {
      step_ += pow(matrix_col_, i);
    }
  } else {
    std::vector<int64_t> cumprod(input_dims_);
    auto dims = input_shape;
    std::partial_sum(dims.begin(), dims.end() - 1, cumprod.begin(), std::multiplies<int64_t>());
    step_ = 1 + std::accumulate(cumprod.begin(), cumprod.end(), static_cast<int64_t>(0));
  }
  if (wrap_ || input_dims_ > kInputMinDim || matrix_row_ < matrix_col_) {
    num_diagonal_elements_ = ceil(static_cast<double>(input_elements_) / step_);
  } else {
    num_diagonal_elements_ = ceil(static_cast<double>(min_size * min_size) / step_);
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
  auto status = CalFillDiagonal(num_diagonal_elements_, fill_value_, step_, output, device_id_,
                                reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, FillDiagonalGpuKernelMod::FillDiagonalFunc>> FillDiagonalGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &FillDiagonalGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &FillDiagonalGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &FillDiagonalGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &FillDiagonalGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &FillDiagonalGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &FillDiagonalGpuKernelMod::LaunchKernel<int16_t>},
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
