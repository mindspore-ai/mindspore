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

#include "plugin/device/gpu/kernel/math/cholesky_inverse_gpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr size_t kZero = 0;
constexpr size_t kOne = 1;
constexpr size_t kDWork = 256;
bool CholeskyInverseGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::CholeskyInverse>(base_operator);
  if (kernel_ptr_ != NULL) {
    kernel_name_ = kernel_ptr_->name();
  }
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' valid gpu kernel does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  upper_ = kernel_ptr_->get_upper();
  if (upper_) {
    uplo_ = CUBLAS_FILL_MODE_LOWER;
  } else {
    uplo_ = CUBLAS_FILL_MODE_UPPER;
  }
  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);

  return true;
}
int CholeskyInverseGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
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
  auto output_shape = outputs[0]->GetShapeVector();
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  std::vector<int64_t> input_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                          inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  int64_t input_dims = input_shape.size();
  if (input_dims <= 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'x' should be at least 1-D, but got " << input_dims
                  << "-D.";
    return false;
  }
  int64_t matrix_row = input_shape[kZero];
  int64_t matrix_col = input_shape[kOne];
  if (matrix_row != matrix_col) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "',input cholesky_inverse must be square matrix "
                  << "while row is" << matrix_row << ", col is" << matrix_col;
  }
  rank_ = matrix_row;
  input_size_list_.push_back(output_elements_ * unit_size_);
  output_size_list_.push_back(output_elements_ * unit_size_);
  workspace_size_list_.push_back(sizeof(int));  // dev_info
  workspace_size_list_.push_back(sizeof(int));  // d_work

  return KRET_OK;
}

template <typename T>
bool CholeskyInverseGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  int *dev_info = GetDeviceAddress<int>(workspace, 0);
  T *d_work = GetDeviceAddress<T>(workspace, 1);
  CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(handle_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                "CusolverDnSetStream failed");

  int lwork = kDWork;

  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSpotri_bufferSize(handle_, uplo_, rank_, input, rank_, &lwork),
                                           "cusolver query spotri work size fail");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDpotri_bufferSize(handle_, uplo_, rank_, input, rank_, &lwork),
                                           "cusolver query dpotri work size fail");
  }

  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnSpotri(handle_, uplo_, rank_, input, rank_, static_cast<T *>(d_work), lwork, dev_info),
      "cusolver dnspotri fail");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnDpotri(handle_, uplo_, rank_, input, rank_, static_cast<T *>(d_work), lwork, dev_info),
      "cussolver dndpotri fail");
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the data type entered must be float or double.";
  }
  if (upper_) {
    CalCopyUpToLow(output_elements_, input, rank_, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else {
    CalCopyLowToUp(output_elements_, input, rank_, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  }
  return True;
}
std::vector<std::pair<KernelAttr, CholeskyInverseGpuKernelMod::CIfunc>> CholeskyInverseGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &CholeskyInverseGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &CholeskyInverseGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> CholeskyInverseGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CIfunc> &pair) { return pair.first; });
  return support_list;
}
}  // namespace kernel
}  // namespace mindspore
